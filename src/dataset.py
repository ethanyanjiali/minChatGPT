from torch.utils.data import Dataset
from datasets import load_dataset
import tiktoken
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer
import torch


class AnthropicHHRLHFDataset(Dataset):
    """
    https://huggingface.co/datasets/Anthropic/hh-rlhf#dataset-summary
    """

    def __init__(self, block_size, split='train', max_examples=None) -> None:
        super().__init__()
        dataset = load_dataset("Anthropic/hh-rlhf", split=split)
        self.pairs = []
        self.masks = []
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        cnt = 0
        for data in dataset:
            positive_indices = encode(data["chosen"])[:block_size]
            positive_mask = [1] * len(positive_indices) + [0] * (
                block_size - len(positive_indices))
            positive_indices += [enc.eot_token
                                 ] * (block_size - len(positive_indices))

            negative_indices = encode(data["rejected"])[:block_size]
            negative_mask = [1] * len(negative_indices) + [0] * (
                block_size - len(negative_indices))
            negative_indices += [enc.eot_token
                                 ] * (block_size - len(negative_indices))

            self.pairs.append(
                torch.tensor((positive_indices, negative_indices),
                             dtype=torch.long))
            self.masks.append(torch.tensor((positive_mask, negative_mask)))
            cnt += 1
            if max_examples and cnt >= max_examples:
                break

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx], self.masks[idx]
