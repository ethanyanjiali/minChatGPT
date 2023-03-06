import torch
import tiktoken
from sentencepiece import SentencePieceProcessor
from typing import List
import os


class TiktokenTokenizer():

    def __init__(self, name) -> None:
        self.enc = tiktoken.get_encoding(name)
        self.encode = lambda s: self.enc.encode(
            s, allowed_special={"<|endoftext|>"})
        self.pad_token = self.enc.eot_token

    def __call__(self,
                 text,
                 max_length=None,
                 padding=None,
                 truncation=False,
                 return_tensors=None):
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


class LLaMATokenizer:
    """
    Copyright (c) Meta Platforms, Inc. and affiliates.
    This software may be used and distributed according to the terms of the GNU General Public License version 3.

    Modified by Ethan Yanjia Li
    - Refactor naming
    - Remove uncessary libraries
    The modification may be used and distributed according to the terms of the GNU General Public License version 3.
    """

    def __init__(self, model_path: str):
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        print(f"Reloaded SentencePiece model from {model_path}")

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        print(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)
