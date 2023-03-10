import functools
import torch
from torch import nn
from torch.utils.data import DataLoader
from loss import KPairwiseLoss, CrossEntropyLoss
import torch.optim as optim
from torch.cuda.amp.grad_scaler import GradScaler
import statistics
from gpt import GPTRewardModel, TransformerDecoderBlock
from tqdm import tqdm, trange
import time
from torch.utils.tensorboard import SummaryWriter
import os
import json
import random
from torchinfo import summary
from configs import TrainingConfig
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
    enable_wrap,
    wrap,
)
# import bitsandbytes as bnb


class Trainer:

    def __init__(self) -> None:
        self.model = None
        self.optimizer = None
        random.seed(1)

    def save_hyperparams(self, hp):
        if not os.path.exists(f'./runs/{self.run_name}'):
            os.makedirs(f'./runs/{self.run_name}')

        with open(f'./runs/{self.run_name}/hyperparams.json', 'w') as fp:
            json.dump(hp, fp, indent=4)

    def save_metrics(self, metrics):
        with open(f'./runs/{self.run_name}/metrics.json', 'w') as fp:
            json.dump(metrics, fp, indent=4)

    def save_states(self, step, is_last=False):
        file_name = f'{self.run_name}_final.pt' if is_last else f'{self.run_name}_step{step}.pt'
        torch.save(
            {
                'step': step,
                'model_state_dict':
                self.model.state_dict(),    # Save the unoptimized model
                'optimizer_state_dict': self.optimizer.state_dict(),
            },
            f'./runs/{self.run_name}/{file_name}')


class SFTTrainer(Trainer):

    def __init__(self, device, model: nn.Module, train_dataset, test_dataset,
                 batch_size, max_steps, cfg, finetune_method) -> None:
        super().__init__()
        self.run_name = f"sft_{str(int(time.time()))}"
        self.device = device
        assert self.device == 'cuda'
        self.max_steps = max_steps
        self.eval_freq = 1
        self.save_freq = 20000
        self.train_dataloader = iter(
            DataLoader(train_dataset,
                       batch_size=batch_size,
                       num_workers=6,
                       pin_memory=True))
        self.test_dataloader = iter(
            DataLoader(test_dataset,
                       batch_size=batch_size,
                       num_workers=6,
                       pin_memory=True))
        self.model = model
        self.criterion = CrossEntropyLoss()

        lr = cfg.lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.grad_clip = 1.0
        self.dtype = torch.float16

        self.finetune_method = finetune_method

        hp = {
            "grad_clip": self.grad_clip,
            "learning_rate": lr,
            "dtype": str(self.dtype),
            "batch_size": batch_size,
            "model": cfg.model_name,
            "lora_rank": model.cfg.lora_rank,
            "block_size": model.cfg.block_size,
            "finetune_method": finetune_method,
            "dropout": model.cfg.dropout_rate
        }
        self.save_hyperparams(hp)

    def fit(self):
        if self.finetune_method:
            self.model.freeze_weights(self.finetune_method)
        summary(self.model, input_data=torch.ones(1, 1024).long())

        opt_model = torch.compile(self.model)
        opt_model.to(self.device)
        writer = SummaryWriter(f'./runs/{self.run_name}/logs', max_queue=40)
        scaler = GradScaler(enabled=self.dtype != torch.float32)

        opt_model.train()
        step = 0

        t0 = time.time()
        while step < self.max_steps:
            x, y = next(self.train_dataloader)
            x = x.to(self.device)
            y = y.to(self.device)

            with torch.autocast(device_type=self.device, dtype=self.dtype):
                y_hat = opt_model(x)    # (B, 1)
                loss = self.criterion(y_hat, y)    # (B, 1)

            if self.grad_clip != 0.0:
                torch.nn.utils.clip_grad_norm_(opt_model.parameters(),
                                               self.grad_clip)

            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            lossf = loss.item()

            iter_time = time.time() - t0
            t0 = time.time()
            print(
                f"step {step}, batch loss {round(lossf,3)}, {round(1.0/iter_time, 2)} iters/s"
            )
            writer.add_scalar('Loss/train/step', lossf, step)

            if step != 0 and step % self.save_freq == 0:
                self.save_states(step)

            step += 1

        self.save_states(step, True)


class RewardModelTrainer(Trainer):

    def __init__(self,
                 cfg: TrainingConfig,
                 device,
                 model: nn.Module,
                 train_dataset,
                 test_dataset,
                 total_epochs,
                 batch_size,
                 finetune_method=False) -> None:
        super().__init__()
        self.run_name = f"rm_{str(int(time.time()))}"
        self.device = device
        assert self.device == 'cuda'
        self.total_epochs = total_epochs
        self.eval_freq = 1
        self.save_freq = 30000
        self.model = model
        self.train_dataloader = DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           num_workers=8,
                                           shuffle=True,
                                           pin_memory=True)
        self.test_dataloader = DataLoader(test_dataset,
                                          batch_size=batch_size,
                                          num_workers=8,
                                          pin_memory=True)
        self.model = model
        self.criterion = KPairwiseLoss()
        self.finetune_method = finetune_method
        lr = 0.0001
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.grad_clip = 1.0
        self.dtype = torch.float16

        hp = {
            "grad_clip": self.grad_clip,
            "learning_rate": lr,
            "dtype": str(self.dtype),
            "batch_size": batch_size,
            "model": cfg.model_name,
            "lora_rank": model.cfg.lora_rank,
            "block_size": model.cfg.block_size,
            "finetune_method": finetune_method,
            "dropout": model.cfg.dropout_rate
        }
        self.save_hyperparams(hp)

    def fit(self):
        if self.finetune_method:
            self.model.freeze_weights(self.finetune_method)
        summary(self.model, input_data=torch.ones(1, 1024).long())

        opt_model = torch.compile(self.model)
        opt_model.to(self.device)
        writer = SummaryWriter(f'./runs/{self.run_name}/logs', max_queue=40)
        scaler = GradScaler(enabled=self.dtype != torch.float32)

        for epoch in range(self.total_epochs):
            opt_model.train()
            for step, (completions, attention_masks) in enumerate(
                    pbar := tqdm(self.train_dataloader)):
                total_steps = step + epoch * len(self.train_dataloader)
                completions = completions.to(self.device)
                attention_masks = attention_masks.to(self.device)

                with torch.autocast(device_type=self.device, dtype=self.dtype):
                    # TODO: Support K completions instead of only 2
                    # TODO: Support gradient accumulation
                    positive_scores = opt_model(
                        completions[:, 0, :],
                        attention_masks[:, 0, :])    # (B, 1)
                    negative_scores = opt_model(
                        completions[:, 1, :],
                        attention_masks[:, 1, :])    # (B, 1)
                    loss = self.criterion(
                        torch.cat((positive_scores, negative_scores),
                                  dim=-1))    # (B, 2)

                if self.grad_clip != 0.0:
                    torch.nn.utils.clip_grad_norm_(opt_model.parameters(),
                                                   self.grad_clip)

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                lossf = loss.item()
                writer.add_scalar('Loss/train/step', lossf, total_steps)
                pbar.set_description(f"batch loss {round(lossf,3)}")

                if total_steps != 0 and total_steps % self.save_freq == 0:
                    self.save_states(total_steps)

            if epoch % self.eval_freq == 0:
                opt_model.eval()
                with torch.no_grad():
                    tp = 0
                    total = 0
                    losses = []
                    for step, (completions, attention_masks) in enumerate(
                            self.test_dataloader):
                        completions = completions.to(self.device)
                        attention_masks = attention_masks.to(self.device)

                        positive_scores = opt_model(
                            completions[:, 0, :],
                            attention_masks[:, 0, :])    # (B, 1)
                        negative_scores = opt_model(
                            completions[:, 1, :],
                            attention_masks[:, 1, :])    # (B, 1)
                        loss = self.criterion(
                            torch.cat((positive_scores, negative_scores),
                                      dim=-1))    # (B, 2)
                        lossf = loss.item()
                        losses.append(lossf)
                        writer.add_scalar(
                            'Loss/test/step', lossf,
                            step + epoch * len(self.test_dataloader))
                        tp += torch.count_nonzero(
                            positive_scores > negative_scores)
                        total += positive_scores.shape[0]

                    acc = tp / total
                    epoch_loss = statistics.mean(losses)

                writer.add_scalar('Loss/test/epoch', epoch_loss, epoch)
                writer.add_scalar('Acc/test/epoch', acc, epoch)
                print(f'Epoch: {epoch+1}, Test Loss: {lossf}, Acc: {acc}')

        self.save_states(total_steps, True)


class FSDPRewardModelTrainer(Trainer):

    def __init__(self,
                 cfg: TrainingConfig,
                 device,
                 model: nn.Module,
                 train_dataset,
                 test_dataset,
                 total_epochs,
                 batch_size,
                 rank,
                 world_size,
                 finetune_method=False) -> None:
        super().__init__()
        self.cfg = cfg
        self.run_name = f"rm_{str(int(time.time()))}"
        self.device = device
        self.rank = rank
        self.world_size = world_size
        assert self.device == 'cuda'
        self.total_epochs = total_epochs
        self.eval_freq = 1
        self.save_freq = 30000
        self.model = model
        self.train_sampler = DistributedSampler(train_dataset,
                                                rank=rank,
                                                num_replicas=world_size,
                                                shuffle=True)
        self.test_sampler = DistributedSampler(test_dataset,
                                               rank=rank,
                                               num_replicas=world_size)
        self.train_dataloader = DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           num_workers=8,
                                           shuffle=True,
                                           pin_memory=True)
        self.test_dataloader = DataLoader(test_dataset,
                                          batch_size=batch_size,
                                          num_workers=8,
                                          pin_memory=True)
        torch.cuda.set_device(rank)

        self.dtype = torch.float16
        self.model = model
        self.criterion = KPairwiseLoss()
        self.finetune_method = finetune_method
        self.optimizer = None    # need to initiatize this later after FSDP
        self.scaler = GradScaler(enabled=self.dtype != torch.float32)
        self.writer = SummaryWriter(f'./runs/{self.run_name}/logs',
                                    max_queue=40)
        self.grad_clip = 1.0

        hp = {
            "grad_clip": self.grad_clip,
            "learning_rate": cfg.lr,
            "dtype": str(self.dtype),
            "batch_size": batch_size,
            "model": cfg.model_name,
            "lora_rank": model.cfg.lora_rank,
            "block_size": model.cfg.block_size,
            "finetune_method": finetune_method,
            "dropout": model.cfg.dropout_rate
        }
        self.save_hyperparams(hp)

    def fit(self):
        if self.finetune_method:
            self.model.freeze_weights(self.finetune_method)
        if self.rank == 0:
            summary(self.model, input_data=torch.ones(1, 1024).long())

        mp = MixedPrecision(param_dtype=self.dtype,
                            reduce_dtype=self.dtype,
                            buffer_dtype=self.dtype)

        model = self.model
        # model = torch.compile(self.model)
        summary(model, input_data=torch.ones(1, 1024).long())
        # model.to(self.rank)

        torch.cuda.set_device(self.rank)
        gpt_auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                TransformerDecoderBlock,
            },
        )
        dist_model = FSDP(
            model,
            mixed_precision=mp,
            use_orig_params=True,
            limit_all_gathers=True,
            auto_wrap_policy=gpt_auto_wrap_policy,
            cpu_offload=CPUOffload(offload_params=True),
            device_id=torch.cuda.current_device(),
            sharding_strategy=ShardingStrategy.FULL_SHARD
            if self.world_size > 1 else ShardingStrategy.NO_SHARD,
        )
        print(dist_model)
        self.optimizer = optim.Adam(dist_model.parameters(), lr=self.cfg.lr)

        for epoch in range(self.total_epochs):
            self.train_epoch(dist_model, epoch)
            if epoch % self.eval_freq == 0:
                self.test_epoch(dist_model, epoch)
            self.save_states(dist_model, epoch)

        test_loss, test_acc = self.test_epoch(dist_model, epoch, False)
        self.save_metrics({"test_loss": test_loss, "test_acc": test_acc})

    def save_states(self, model, epoch=None):
        if self.rank == 0:
            save_policy = FullStateDictConfig(
            # offload_to_cpu=True and NO_SHARD is not supported yet
                offload_to_cpu=self.world_size > 1,
                rank0_only=True)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                                      save_policy):
                cpu_state = model.state_dict()
                # remove the _orig_mod prefix from torch.compile
                cpu_state = {
                    k.partition("_orig_mod.")[2]: cpu_state[k]
                    for k in cpu_state.keys()
                }
                file_name = f'./runs/{self.run_name}/{self.run_name}_{epoch if epoch else "final"}.pt'
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': cpu_state,
                        'optimizer_state_dict': self.optimizer.state_dict(),
                    }, file_name)

    def train_epoch(self, model, epoch, logging=True):
        self.train_sampler.set_epoch(epoch)
        model.train()
        if self.rank == 0:
            pbar = tqdm(range(len(self.train_dataloader)), colour="blue")

        epoch_data = torch.zeros(3).to(self.rank)    # [loss, tp, total]

        for step, data in enumerate(self.train_dataloader):
            completions, attention_masks = data
            total_steps = step + epoch * len(self.train_dataloader)
            completions = completions.to(self.rank)
            attention_masks = attention_masks.to(self.rank)

            with torch.autocast(device_type=self.device, dtype=self.dtype):
                # TODO: Support K completions instead of only 2
                # TODO: Support gradient accumulation
                positive_scores = model(completions[:, 0, :],
                                        attention_masks[:, 0, :])    # (B, 1)
                negative_scores = model(completions[:, 1, :],
                                        attention_masks[:, 1, :])    # (B, 1)
                loss = self.criterion(
                    torch.cat((positive_scores, negative_scores),
                              dim=-1))    # (B, 2)

            if self.grad_clip != 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               self.grad_clip)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.optimizer.zero_grad(set_to_none=True)

            lossf = loss.item()
            epoch_data[0] += lossf
            epoch_data[1] += torch.count_nonzero(
                positive_scores > negative_scores)
            epoch_data[2] += positive_scores.shape[0]

            if logging:
                self.writer.add_scalar(f'Loss/train/step/{self.rank}', lossf,
                                       total_steps)
            if self.rank == 0:
                pbar.update(1)
                pbar.set_description(
                    f"Epoch {epoch}, train loss {round(lossf,3)}")

        dist.all_reduce(epoch_data, op=dist.ReduceOp.SUM)
        train_acc = epoch_data[1].item() / epoch_data[2].item()
        train_loss = epoch_data[0].item() / len(self.train_dataloader)

        if self.rank == 0:
            pbar.close()
            print(
                f"Epoch {epoch}, avg train loss {round(train_loss,3)}, train acc {round(train_acc,3)}"
            )
            if logging:
                self.writer.add_scalar(f'Loss/train/epoch', train_loss, epoch)
                self.writer.add_scalar(f'Acc/train/epoch', train_acc, epoch)

        return train_loss

    @torch.no_grad()
    def test_epoch(self, model, epoch, logging=True):
        model.eval()
        if self.rank == 0:
            pbar = tqdm(range(len(self.test_dataloader)), colour="blue")

        epoch_data = torch.zeros(3).to(self.rank)    # [loss, tp, total]
        for step, data in enumerate(self.test_dataloader):
            total_steps = step + epoch * len(self.test_dataloader)
            completions, attention_masks = data
            completions = completions.to(self.device)
            attention_masks = attention_masks.to(self.device)

            positive_scores = model(completions[:, 0, :],
                                    attention_masks[:, 0, :])    # (B, 1)
            negative_scores = model(completions[:, 1, :],
                                    attention_masks[:, 1, :])    # (B, 1)
            loss = self.criterion(
                torch.cat((positive_scores, negative_scores),
                          dim=-1))    # (B, 2)

            lossf = loss.item()
            epoch_data[0] += lossf
            epoch_data[1] += torch.count_nonzero(
                positive_scores > negative_scores)
            epoch_data[2] += positive_scores.shape[0]

            if logging:
                self.writer.add_scalar(f'Loss/test/step/{self.rank}', lossf,
                                       total_steps)

            if self.rank == 0:
                pbar.update(1)
                pbar.set_description(
                    f"Epoch {epoch}, test loss {round(lossf,3)}")

        dist.all_reduce(epoch_data, op=dist.ReduceOp.SUM)
        test_acc = epoch_data[1].item() / epoch_data[2].item()
        test_loss = epoch_data[0].item() / len(self.test_dataloader)

        if self.rank == 0:
            pbar.close()
            print(
                f"Epoch {epoch}, avg test loss {round(test_loss,3)}, test acc {round(test_acc,3)}"
            )
            if logging:
                self.writer.add_scalar(f'Loss/test/epoch', test_loss, epoch)
                self.writer.add_scalar(f'Acc/test/epoch', test_acc, epoch)

        return test_loss, test_acc
