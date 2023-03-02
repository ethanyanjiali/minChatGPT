from models import GPT, GPTRewardModel, HFGPTRewardModel
from dataset import AnthropicHHRLHFDataset
import torch
import tiktoken
import click
from torch.utils.data import DataLoader
from trainers import RewardModelTrainer
from torchinfo import summary


@click.command()
@click.option('--task', '-t')
def main(task):
    device = 'cuda'
    max_new_tokens = 20
    num_samples = 8
    temperature = 0.9
    top_k = 200
    prompt = "Hello, my name is"
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)
    indices = encode(prompt)
    x = (torch.tensor(indices, dtype=torch.long, device=device)[None, ...])

    if task == 'gpt':
        model = GPT.from_pretrained()
        model.eval()
        model.to(device)
        for k in range(num_samples):
            y = model.generate(x,
                               max_new_tokens,
                               temperature=temperature,
                               top_k=top_k)
            print(decode(y[0].tolist()))
            print('---------------')
    elif task == "reward":
        rm = GPTRewardModel.from_pretrained('gpt2-xl')
        rm.eval()
        rm.to(device)
        score = rm(x)
        print(score)
    elif task == "dataset":
        from datasets import load_dataset
        dataset = load_dataset("Anthropic/hh-rlhf", split='train')
        print(dataset[0])
    elif task == "test_dataset":
        model = GPT.from_pretrained('gpt2-large')
        model.eval()
        model.to(device)
        ds = AnthropicHHRLHFDataset()
        dl = DataLoader(ds, batch_size=1)
        for d in dl:
            print('Positive---------------')
            x = d[0].to(device)
            y = model.generate(x,
                               max_new_tokens,
                               temperature=temperature,
                               top_k=top_k)
            print(decode(y[0].tolist()))
            print('Negative---------------')
            x = d[1].to(device)
            y = model.generate(x,
                               max_new_tokens,
                               temperature=temperature,
                               top_k=top_k)
            print(decode(y[0].tolist()))
            print('End---------------')
    elif task == "train_rm":
        rm = GPTRewardModel.from_pretrained('gpt2-xl')
        rm.freeze_weights()
        summary(rm, input_data=torch.ones(1, 1024).long())
        train_ds = AnthropicHHRLHFDataset(block_size=1024,
                                          split='train',
                                          max_examples=None,
                                          tokenizer_name="tiktoken/gpt2")
        test_ds = AnthropicHHRLHFDataset(block_size=1024,
                                         split='test',
                                         max_examples=None,
                                         tokenizer_name="tiktoken/gpt2")
        trainer = RewardModelTrainer(device,
                                     rm,
                                     train_ds,
                                     test_ds,
                                     total_epochs=10)
        trainer.fit()


if __name__ == "__main__":
    main()
