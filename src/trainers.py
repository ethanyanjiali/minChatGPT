import torch
from torch import nn
from torch.utils.data import DataLoader
from loss import KPairwiseLoss
import torch.optim as optim
from torch.cuda.amp.grad_scaler import GradScaler
import statistics
from tqdm import tqdm, trange
import time
from torch.utils.tensorboard import SummaryWriter
import os


class Trainer:

    def __init__(self) -> None:
        pass


class RewardModelTrainer(Trainer):

    def __init__(self, device, model: nn.Module, train_dataset, test_dataset,
                 total_epochs) -> None:
        super().__init__()
        self.run_name = int(time.time())
        self.device = device
        assert self.device == 'cuda'
        self.total_epochs = total_epochs
        self.eval_freq = 1
        self.model = model
        self.train_dataloader = DataLoader(train_dataset,
                                           batch_size=1,
                                           num_workers=6)
        self.test_dataloader = DataLoader(test_dataset,
                                          batch_size=1,
                                          num_workers=6)
        self.model = model
        self.criterion = KPairwiseLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.grad_clip = 1.0
        self.dtype = torch.float16

    def fit(self):
        self.model = torch.compile(self.model)
        self.model.to(self.device)
        writer = SummaryWriter(f'./logs/{self.run_name}', max_queue=20)
        scaler = GradScaler(enabled=self.dtype != torch.float32)

        for epoch in range(self.total_epochs):
            losses = []
            self.model.train()
            for step, (completions, attention_masks) in enumerate(
                    tqdm(self.train_dataloader)):

                completions = completions.to(self.device)
                attention_masks = attention_masks.to(self.device)

                with torch.autocast(device_type=self.device, dtype=self.dtype):
                    # TODO: Support K completions instead of only 2
                    # TODO: Support gradient accumulation
                    positive_scores = self.model(
                        completions[:, 0, :],
                        attention_masks[:, 0, :])    # (B, 1)
                    negative_scores = self.model(
                        completions[:, 1, :],
                        attention_masks[:, 1, :])    # (B, 1)
                    loss = self.criterion(
                        torch.cat((positive_scores, negative_scores),
                                  dim=-1))    # (B, 2)

                if self.grad_clip != 0.0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   self.grad_clip)

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                lossf = loss.item()
                writer.add_scalar('Loss/train/step', lossf,
                                  step + epoch * len(self.train_dataloader))
                losses.append(lossf)
                # print(f'Step: {step+1}, Loss: {loss}')

            epoch_loss = statistics.mean(losses)
            writer.add_scalar('Loss/train/epoch', epoch_loss, epoch)
            print(f'Epoch: {epoch+1}, Train Loss: {epoch_loss}')

            if epoch % self.eval_freq == 0:
                self.model.eval()
                with torch.no_grad():
                    tp = 0
                    total = 0
                    losses = []
                    for step, (completions, attention_masks) in enumerate(
                            self.test_dataloader):
                        completions = completions.to(self.device)
                        attention_masks = attention_masks.to(self.device)

                        positive_scores = self.model(
                            completions[:, 0, :],
                            attention_masks[:, 0, :])    # (B, 1)
                        negative_scores = self.model(
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

                if not os.path.exists('./weights/'):
                    os.makedirs('./weights')
                    torch.save(
                        {
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict':
                            self.optimizer.state_dict(),
                        }, f'./weights/{self.run_name}_epoch{epoch}.pt')
