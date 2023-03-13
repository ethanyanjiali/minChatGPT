import click
import torch
from trainers import PPOTrainer
from configs import get_configs
from gpt import GPTActor, GPTRewardModel, GPTCritic
from dataset import DahoasSFTStaticPromptsDataset


def train():
    cfg = get_configs("gpt2-medium")
    actor = GPTActor.from_checkpoint(
        cfg,
        "./runs/sft_1678085469/original_sft_1678085469_step100000.pt").cuda()
    critic = GPTCritic.from_checkpoint(
        cfg, "./runs/rm_1678145909/rm_1678145909_final.pt").cuda()
    sft_model = GPTActor.from_checkpoint(
        cfg,
        "./runs/sft_1678085469/original_sft_1678085469_step100000.pt").cuda()
    reward_model = GPTRewardModel.from_checkpoint(
        cfg, "./runs/rm_1678145909/rm_1678145909_final.pt").cuda()

    dataset = DahoasSFTStaticPromptsDataset(block_size=1024,
                                            max_examples=10,
                                            tokenizer_name="tiktoken/gpt2")
    trainer = PPOTrainer(cfg,
                         actor,
                         critic,
                         reward_model,
                         sft_model,
                         dataset,
                         batch_size=2)
    trainer.fit()
    # idx, masks = next(iter(trainer.train_dataloader))
    # trainer.make_experience(idx.cuda(), masks.cuda())


@click.command()
@click.option('--strategy', '-s')
def main(strategy):
    train()


if __name__ == "__main__":
    main()
