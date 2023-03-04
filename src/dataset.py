from torch.utils.data import Dataset, IterableDataset
from datasets import load_dataset
import tiktoken
from transformers import GPT2Tokenizer, GPT2TokenizerFast
import torch
from tqdm import tqdm
import random


class TiktokenTokenizer():

    def __init__(self, name) -> None:
        self.enc = tiktoken.get_encoding(name)
        self.encode = lambda s: self.enc.encode(
            s, allowed_special={"<|endoftext|>"})
        self.pad_token = self.enc.eot_token

    def __call__(self,
                 text,
                 max_length=None,
                 padding="max_length",
                 truncation=True,
                 return_tensors="pt"):
        ids = self.encode(text)
        if truncation:
            ids = ids[:max_length]
        mask = [1] * len(ids)
        if padding == "max_length":
            mask += [0] * (max_length - len(ids))
            ids += [self.pad_token] * (max_length - len(ids))

        if return_tensors == "pt":
            ids = torch.tensor(ids, dtype=torch.long)
            mask = torch.tensor(mask)

        return {"input_ids": ids, "attention_mask": mask}


class DahoasSFTStaticDataset(IterableDataset):
    """
    https://huggingface.co/datasets/Dahoas/sft-static
    """

    def __init__(self,
                 block_size,
                 split='train',
                 max_examples=None,
                 tokenizer_name='tiktoken/gpt2') -> None:
        super().__init__()
        dataset = load_dataset("Dahoas/sft-static", split=split)
        self.tokens = []
        self.block_size = block_size

        if tokenizer_name == "huggingface/gpt2":
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer_name == "huggingface/gpt2fast":
            tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        elif tokenizer_name == "tiktoken/gpt2":
            tokenizer = TiktokenTokenizer('gpt2')

        cnt = 0
        for data in tqdm(dataset):
            cnt += 1
            prompt = data['prompt']

            response_text += prompt + data['response'] + "<|endoftext|>"
            response = tokenizer(response_text)

            self.tokens += response['input_ids']
            if max_examples and cnt >= max_examples:
                break

        self.tokens = torch.tensor(self.tokens, dtype=torch.long)

    def __iter__(self):
        start = random.randint(0, len(self.tokens) - self.block_size - 2)
        x = self.tokens[start:start + self.block_size]
        y = self.tokens[start + 1:start + self.block_size + 1]
        yield x, y


class DahoasRMStaticDataset(Dataset):
    """
    https://huggingface.co/datasets/Dahoas/rm-static
    """

    def __init__(self,
                 block_size,
                 split='train',
                 max_examples=None,
                 tokenizer_name='tiktoken/gpt2') -> None:
        super().__init__()
        dataset = load_dataset("Dahoas/rm-static", split=split)
        self.pairs = []
        self.masks = []

        if tokenizer_name == "huggingface/gpt2":
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer_name == "huggingface/gpt2fast":
            tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        elif tokenizer_name == "tiktoken/gpt2":
            tokenizer = TiktokenTokenizer('gpt2')

        cnt = 0
        for data in tqdm(dataset):
            cnt += 1
            prompt = data['prompt']

            positive_text = prompt + data['chosen'] + "<|endoftext|>"
            positive = tokenizer(positive_text,
                                 max_length=block_size,
                                 padding="max_length",
                                 truncation=True,
                                 return_tensors="pt")

            negative_text = prompt + data['rejected'] + "<|endoftext|>"
            negative = tokenizer(negative_text,
                                 max_length=block_size,
                                 padding="max_length",
                                 truncation=True,
                                 return_tensors="pt")

            self.pairs.append(
                torch.stack((positive['input_ids'], negative['input_ids']),
                            dim=0))

            self.masks.append(
                torch.stack(
                    (positive['attention_mask'], negative['attention_mask']),
                    dim=0))
            if max_examples and cnt >= max_examples:
                break

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx], self.masks[idx]    # (2, T), (2, T)


class AnthropicHHRLHFDataset(Dataset):
    """
    https://huggingface.co/datasets/Anthropic/hh-rlhf#dataset-summary
    """

    def __init__(self,
                 block_size,
                 split='train',
                 max_examples=None,
                 tokenizer_name='tiktoken/gpt2') -> None:
        super().__init__()
        dataset = load_dataset("Anthropic/hh-rlhf", split=split)
        self.pairs = []
        self.masks = []

        if tokenizer_name == "huggingface/gpt2":
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer_name == "huggingface/gpt2fast":
            tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        elif tokenizer_name == "tiktoken/gpt2":
            tokenizer = TiktokenTokenizer('gpt2')

        cnt = 0
        for data in dataset:
            positive = tokenizer(data["chosen"],
                                 max_length=block_size,
                                 padding="max_length",
                                 truncation=True,
                                 return_tensors="pt")
            positive_indices = positive["input_ids"]
            positive_mask = positive["attention_mask"]

            negative = tokenizer(data["rejected"],
                                 max_length=block_size,
                                 padding="max_length",
                                 truncation=True,
                                 return_tensors="pt")
            negative_indices = negative["input_ids"]
            negative_mask = negative["attention_mask"]

            self.pairs.append(
                torch.stack((positive_indices, negative_indices), dim=0))

            self.masks.append(
                torch.stack((positive_mask, negative_mask), dim=0))
            cnt += 1
            if max_examples and cnt >= max_examples:
                break

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx], self.masks[idx]    # (2, T), (2, T)
