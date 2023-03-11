import click
from trainers import PPOTrainer
from configs import get_configs
from gpt import GPT, GPTRewardModel
from dataset import AwesomeChatGPTPromptsDataset


def train():
    cfg = get_configs("gpt2-medium")
    actor = GPT.from_checkpoint(
        cfg, "./runs/sft_1678085469/original_sft_1678085469_step100000.pt")
    critic = GPTRewardModel.from_checkpoint(
        cfg, "./runs/rm_1678145909/rm_1678145909_final.pt")
    sft_model = GPT.from_checkpoint(
        cfg, "./runs/sft_1678085469/original_sft_1678085469_step100000.pt")
    reward_model = GPTRewardModel.from_checkpoint(
        cfg, "./runs/rm_1678145909/rm_1678145909_final.pt")

    dataset = AwesomeChatGPTPromptsDataset(block_size=1024,
                                           tokenizer_name="tiktoken/gpt2")
    trainer = PPOTrainer(cfg,
                         actor,
                         critic,
                         reward_model,
                         sft_model,
                         dataset,
                         batch_size=2)
    trainer.fit()


@click.command()
@click.option('--strategy', '-s')
def main(strategy):
    train()


if __name__ == "__main__":
    main()
