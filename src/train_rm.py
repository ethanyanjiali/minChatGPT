import os
import click
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from trainers import RewardModelTrainer, FSDPRewardModelTrainer
from configs import get_configs
from gpt import GPTRewardModel
from dataset import DahoasRMStaticDataset


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def train_fsdp(rank, world_size, pretrain):
    print(f"Start rank {rank} with world size {world_size}")
    setup(rank, world_size)
    device = "cuda"
    cfg = get_configs("gpt2-xl")
    rm = GPTRewardModel.from_pretrained(cfg)
    train_ds = DahoasRMStaticDataset(block_size=1024,
                                     split='train',
                                     max_examples=20,
                                     tokenizer_name="tiktoken/gpt2")
    test_ds = DahoasRMStaticDataset(block_size=1024,
                                    split='test',
                                    max_examples=20,
                                    tokenizer_name="tiktoken/gpt2")
    trainer = FSDPRewardModelTrainer(cfg,
                                     device,
                                     rm,
                                     train_ds,
                                     test_ds,
                                     total_epochs=1,
                                     batch_size=1,
                                     rank=rank,
                                     world_size=world_size,
                                     finetune_method=False)
    trainer.fit()
    dist.barrier()
    cleanup()


def train(pretrain):
    device = 'cuda'
    cfg = get_configs("gpt2-medium/lora")
    if pretrain == "gpt2":
        rm = GPTRewardModel.from_pretrained(cfg)
    else:
        rm = GPTRewardModel.from_backbone_checkpoint(
            cfg, "./runs/sft_1678085469/original_sft_1678085469_step100000.pt")
    train_ds = DahoasRMStaticDataset(block_size=1024,
                                     split='train',
                                     max_examples=20,
                                     tokenizer_name="tiktoken/gpt2")
    test_ds = DahoasRMStaticDataset(block_size=1024,
                                    split='test',
                                    max_examples=20,
                                    tokenizer_name="tiktoken/gpt2")
    trainer = RewardModelTrainer(cfg,
                                 device,
                                 rm,
                                 train_ds,
                                 test_ds,
                                 batch_size=1,
                                 total_epochs=1,
                                 finetune_method="lora")
    trainer.fit()


@click.command()
@click.option('--mode', '-m')
@click.option('--pretrain', '-p')
def main(mode, pretrain):
    torch.manual_seed(1234)

    if mode == "fsdp":
        WORLD_SIZE = torch.cuda.device_count()
        mp.spawn(train_fsdp,
                 args=(WORLD_SIZE, pretrain),
                 nprocs=WORLD_SIZE,
                 join=True)
    elif mode == "naive":
        train(pretrain)


if __name__ == "__main__":
    main()
