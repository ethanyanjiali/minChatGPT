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
        self.qkv_projection = nn.Linear(cfg.embedding_dim,
                                        3 * cfg.embedding_dim,
                                        bias=cfg.use_bias)
        self.output_projection = nn.Linear(cfg.embedding_dim,
                                           cfg.embedding_dim,
                                           bias=cfg.use_bias)
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
        self.register_buffer("mask", mask)

    def forward(self, x: Tensor):
        """
        x: shape of (B, T, C)
        """
        B, T, C = x.size()
        # Project x three times and split into Q,K,V
        x3 = self.qkv_projection(x)    # (B, T, 3C)
        Q, K, V = x3.split(self.cfg.embedding_dim,
                           dim=2)    # (B, T, C) for each

        # Prepare Q,K,V into desired shape for multi-head attention
        # Multi-head attention is equivalent to single-head attention on sequence-tensor form
        # see 3.1 in [3]
        Q = Q.view(B, T, self.cfg.n_heads,
                   C // self.cfg.n_heads)    # (B, T, h, h_dim)
        Q = Q.transpose(1, 2)    # (B, h, T, h_dim)
        K = K.view(B, T, self.cfg.n_heads,
                   C // self.cfg.n_heads)    # (B, T, h, h_dim)
        K = K.transpose(1, 2)    # (B, h, T, h_dim)
        V = V.view(B, T, self.cfg.n_heads,
                   C // self.cfg.n_heads)    # (B, T, h, h_dim)
        V = V.transpose(1, 2)    # (B, h, T, h_dim)

        # (B, h, T, h_dim) @ (B, h, h_dim, T) -> (B, h, T, T)
        attention = Q @ K.transpose(2, 3)
        attention *= 1.0 / math.sqrt(K.size(-1))
        # In transformer decoder, one word can only attend to words before itself
        # also, we don't need the full mask, just need one with shape of (1, 1, T, T)
        attention = attention.masked_fill(self.mask[:, :, :T, :T] == 0,
                                          float('-inf'))    # (B, h, T, T)

        attention = F.softmax(attention, dim=-1)    # (B, h, T, T)
        attention = self.attention_dropout(attention)
        # (B, h, T, T) @ (B, h, T, h_dim) -> (B, h, T, h_dim)
        # V weighted by attention
        weighted_value = attention @ V
        # restore the original shape (B, T, C)
        weighted_value = weighted_value.transpose(1, 2).contiguous().view(
            B, T, C)

        # Finally, linearly project the weighted value to get the output
        y = self.output_projection(weighted_value)
        y = self.output_dropout(y)
        return y


class FeedForwardNetworks(nn.Module):

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(cfg.embedding_dim,
                             4 * cfg.embedding_dim,
                             bias=cfg.use_bias)
        self.fc2 = nn.Linear(4 * cfg.embedding_dim,
                             cfg.embedding_dim,
                             bias=cfg.use_bias)
        self.dropout = nn.Dropout(cfg.dropout_rate)

    def gelu(self, x):
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

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
        self.ln1 = nn.LayerNorm(cfg.embedding_dim,
                                elementwise_affine=cfg.use_bias)
        self.mmsa = MaskedMultiheadSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.embedding_dim,
                                elementwise_affine=cfg.use_bias)
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

        self.transformer = nn.ModuleDict({
            "token_embedding_layer":
            nn.Embedding(cfg.vocab_size, cfg.embedding_dim),    # (Vocab, d)
            "postion_embedding_layer":
            nn.Embedding(cfg.block_size, cfg.embedding_dim),
        # "We apply dropout tp the sums of the embeddings and the positional encodings"
        # 5.4 in [1]
            "input_dropout":
            nn.Dropout(cfg.dropout_rate),
            "decoder_blocks":
            nn.ModuleList(
                [TransformerDecoderBlock(cfg) for _ in range(cfg.n_layers)]),
            "ln":
            nn.LayerNorm(cfg.embedding_dim, elementwise_affine=cfg.use_bias)
        })

        # Final linear layer w/o softmax
        self.transformer_head = nn.Linear(cfg.embedding_dim,
                                          cfg.vocab_size,
                                          bias=False)

    def forward(self, indices: Tensor):
        """
        indices: Shape of (B, T)
        """
        B, T = indices.size()
        pos = torch.arange(0, T, dtype=torch.long,
                           device=indices.device).unsqueeze(0)    # (1, T)
        token_embeddings = self.transformer.token_embedding_layer(
            indices)    # (B, T, d)
        pos_embeddings = self.transformer.postion_embedding_layer(
            pos)    # (B, T, d)
        x = self.transformer.input_dropout(token_embeddings + pos_embeddings)

        # N decoder blocks
        for block in self.transformer.decoder_blocks:
            x = block(x)

        x = self.transformer.ln(x)
        logits = self.transformer_head(x)

        return logits

    @classmethod
    def from_pretrained(cls):
        """
        https://github.com/karpathy/nanoGPT/blob/master/model.py#L213
        """

        def convert_state_key(k):
            huggingface_names = {
                "token_embedding_layer": "wte",
                "postion_embedding_layer": "wpe",
                "decoder_blocks": "h",
                "mmsa": "attn",
                "ln1": "ln_1",
                "ln2": "ln_2",
                "ffn": "mlp",
                "fc1": "c_fc",
                "fc2": "c_proj",
                "qkv_projection": "c_attn",
                "output_projection": "c_proj",
                "ln": "ln_f",
                "transformer_head": "lm_head",
            }
            hf_key = []
            for name in k.split('.'):
                hf_key.append(huggingface_names.get(name, name))
            return '.'.join(hf_key)

        def should_transpose(k):
            transposed = [
                'attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight',
                'mlp.c_proj.weight'
            ]
            for t in transposed:
                if k.endswith(t):
                    return True
            return False

        cfg = GPTConfig(
            n_layers=48,
            n_heads=25,
            embedding_dim=1600,
            dropout_rate=0,
            use_bias=True,
            block_size=1024,
            vocab_size=50257,
        )
        model = GPT(cfg)
        model_states = model.state_dict()
        model_states_keys = [
            k for k in model_states.keys() if not k.endswith('.mmsa.mask')
        ]

        from transformers import GPT2LMHeadModel
        model_pretrained = GPT2LMHeadModel.from_pretrained('gpt2-xl')
        pretrained_states = model_pretrained.state_dict()

        pretrained_states_keys = [
            k for k in pretrained_states.keys()
            if not k.endswith('.attn.masked_bias')
        ]
        pretrained_states_keys = [
            k for k in pretrained_states_keys if not k.endswith('.attn.bias')
        ]

        assert len(pretrained_states_keys) == len(
            model_states_keys
        ), f"mismatched keys: {len(pretrained_states_keys)} != {len(model_states_keys)}"

        for dst_key in model_states_keys:
            src_key = convert_state_key(dst_key)
            if should_transpose(src_key):
                assert pretrained_states[src_key].shape[::-1] == model_states[
                    dst_key].shape
                with torch.no_grad():
                    model_states[dst_key].copy_(pretrained_states[src_key].t())
            else:
                assert pretrained_states[src_key].shape == model_states[
                    dst_key].shape
                with torch.no_grad():
                    model_states[dst_key].copy_(pretrained_states[src_key])

        return model

    @torch.no_grad()
    def generate(self, indices, max_new_tokens, temperature=1.0, top_k=None):
        """
        https://github.com/karpathy/nanoGPT/blob/master/model.py#L343
    
        Take a conditioning sequence of indices (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            indices_cond = indices if indices.size(
                1) <= self.cfg.block_size else indices[:,
                                                       -self.cfg.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits = self(indices_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            indices_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            indices = torch.cat((indices, indices_next), dim=1)

        return indices
