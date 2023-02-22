import math
import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F

from configs import GPTConfig

# [1] Attention is all you need
# [2] Improving Language Understanding by Generated Pre-Training
# [3] Note 10: Self-Attention & Transformers


class MaskedMultiheadSelfAttention(nn.Module):

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        # Figure 2 in [1]
        self.cfg: GPTConfig = cfg
        self.qkv_projection = nn.Linear(
            cfg.embedding_dim, 3 * cfg.embedding_dim, bias=cfg.use_bias)
        self.output_projection = nn.Linear(
            cfg.embedding_dim, cfg.embedding_dim, bias=cfg.use_bias)
        self.attention_dropout = nn.Dropout(cfg.dropout_rate)
        self.output_dropout = nn.Dropout(cfg.dropout_rate)

        # construct a mask like this
        # [[1, 0, 0]
        #  [1, 1, 0]]
        #  [1, 1, 1]] when block_size is 3
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        # insert (B, T) dimension for broadcasting later
        mask = mask.view(1, 1, cfg.block_size, cfg.block_size)
        # mask is a constant and shouldn't be considered as parameters
        # (1, 1, block_size, block_size)
        self.mask = self.register_buffer("mask", mask)

    def forward(self, x: Tensor):
        """
        x: shape of (B, T, C)
        """
        B, T, C = x.size()
        # Project x three times and split into Q,K,V
        x3 = self.qkv_projection(x)  # (B, T, 3C)
        Q, K, V = x3.split(self.cfg.embedding_dim, dim=2)  # (B, T, C) for each

        # Prepare Q,K,V into desired shape for multi-head attention
        # Multi-head attention is equivalent to single-head attention on sequence-tensor form
        # see 3.1 in [3]
        Q = Q.view(B, T, self.cfg.n_head, C //
                   self.cfg.n_head)  # (B, T, h, h_dim)
        Q = Q.transpose(1, 2)  # (B, h, T, h_dim)
        K = K.view(B, T, self.cfg.n_head, C //
                   self.cfg.n_head)  # (B, T, h, h_dim)
        K = K.transpose(1, 2)  # (B, h, T, h_dim)
        V = V.view(B, T, self.cfg.n_head, C //
                   self.cfg.n_head)  # (B, T, h, h_dim)
        V = V.transpose(1, 2)  # (B, h, T, h_dim)

        # (B, h, T, h_dim) @ (B, h, h_dim, T) -> (B, h, T, T)
        attention = Q @ K.transpose(2, 3)
        attention *= 1.0 / math.sqrt(K.size(-1))
        # In transformer decoder, one word can only attend to words before itself
        # also, we don't need the full mask, just need one with shape of (1, 1, T, T)
        attention = attention.masked_fill(
            self.bias[:, :, :T, :T] == 0, float('-inf'))  # (B, h, T, T)

        attention = F.softmax(attention, dim=-1)  # (B, h, T, T)
        attention = self.attention_dropout(attention)
        # (B, h, T, T) @ (B, h, T, h_dim) -> (B, h, T, h_dim)
        # V weighted by attention
        wegithed_value = attention @ V
        # restore the original shape (B, T, C)
        wegithed_value = wegithed_value.transpose(1, 2).view(B, T, C)

        # Finally, linearly project the weighted value to get the output
        y = self.output_projection(wegithed_value)
        y = self.output_dropout(y)
        return y


class FeedForwardNetworks(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(cfg.embedding_dim, 4 *
                             cfg.embedding_dim, bias=cfg.use_bias)
        self.fc2 = nn.Linear(4 * cfg.embedding_dim,
                             cfg.embedding_dim, bias=cfg.use_bias)
        self.dropout = nn.Dropout(cfg.dropout_rate)

    def gelu(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        y = self.dropout(x)
        return y


class TransformerDecoderBlock(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg: GPTConfig = cfg
        self.ln1 = nn.LayerNorm(cfg.embedding_dim, bias=cfg.use_bias)
        self.mmsa = MaskedMultiheadSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.embedding_dim, bias=cfg.use_bias)
        self.ffn = FeedForwardNetworks(cfg)

    def forward(self, x):
        identity1 = x
        x = self.ln1(x)
        x = self.mmsa(x)
        x = identity1 + x

        identity2 = x
        x = self.ln2(x)
        x = self.ffn(x)
        y = identity2 + x
        return y


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg: GPTConfig = cfg
        self.token_embedding_layer = nn.Embedding(
            cfg.vocab_size, cfg.embedding_dim)  # (Vocab, d)
        self.postion_embedding_layer = nn.Embedding(
            cfg.block_size, cfg.embedding_dim)
        # "We apply dropout tp the sums of the embeddings and the positional encodings"
        # 5.4 in [1]
        self.input_dropout = nn.Dropout(cfg.dropout_rate)
        self.decoder_blocks = nn.ModuleList(
            [TransformerDecoderBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln = nn.LayerNorm(cfg.embedding_dim, bias=cfg.use_bias)
        # Final linear layer w/o softmax
        self.transformer_head = nn.Linear(cfg.embedding_dim, bias=cfg.use_bias)

    def forward(self, indices: Tensor):
        """
        indices: Shape of (B, T)
        """
        B, T = indices.size()
        pos = torch.arange(0, T, dtype=torch.long,
                           device=indices.device()).unsqueeze(0)  # (1, T)
        token_embeddings = self.token_embedding_layer(indices)  # (B, T, d)
        pos_embeddings = self.postion_embedding_layer(pos)  # (B, T, d)
        x = self.input_dropout(token_embeddings + pos_embeddings)

        # N decoder blocks
        for block in self.decoder_blocks:
            x = block(x)

        x = self.ln(x)
        logits = self.transformer_head(x)

        return logits
