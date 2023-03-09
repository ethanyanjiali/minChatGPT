import click
import torch
from trainers import SFTTrainer
from gpt import GPT
from dataset import EYLSFTStaticDataset
from configs import get_configs


def train():
    device = 'cuda'
    cfg = get_configs("gpt2-xl/dropout")
    model = GPT.from_pretrained(cfg)
    train_ds = EYLSFTStaticDataset(block_size=1024,
                                   split='train',
                                   max_examples=None,
                                   tokenizer_name="tiktoken/gpt2")
    test_ds = EYLSFTStaticDataset(block_size=1024,
                                  split='test',
                                  max_examples=None,
                                  tokenizer_name="tiktoken/gpt2")
    trainer = SFTTrainer(cfg,
                         device,
                         model,
                         train_ds,
                         test_ds,
                         batch_size=2,
                         max_steps=100001,
                         finetune_method=False)


@click.command()
@click.option('--strategy', '-s')
def main(strategy, pretrain):
    torch.manual_seed(1234)
    train(pretrain)


if __name__ == "__main__":
    main()
