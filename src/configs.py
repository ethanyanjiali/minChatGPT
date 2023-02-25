from dataclasses import dataclass


@dataclass
class GPTConfig:
    n_layers: int
    n_heads: int
    embedding_dim: int
    dropout_rate: float
    use_bias: bool
    block_size: int
    vocab_size: int
