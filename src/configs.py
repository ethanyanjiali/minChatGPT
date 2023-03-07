from dataclasses import dataclass


@dataclass
class TrainingConfig:
    n_layers: int
    n_heads: int
    embedding_dim: int
    dropout_rate: float
    use_bias: bool
    block_size: int
    vocab_size: int
    model_name: str
    hf_model: str
    lr: float = 0.0001
    lora_rank: int = 0
    pretrain: str = "huggingface"


def get_configs(name):
    if name == "gpt2-medium":
        return TrainingConfig(
            n_layers=24,
            n_heads=16,
            embedding_dim=1024,
            dropout_rate=0,
            use_bias=True,
            block_size=1024,
            vocab_size=50257,
            model_name="gpt2-medium",
            hf_model="gpt2-medium",
        )
    elif name == "gpt2-medium/dropout":
        return TrainingConfig(
            n_layers=24,
            n_heads=16,
            embedding_dim=1024,
            dropout_rate=0.2,
            use_bias=True,
            block_size=1024,
            vocab_size=50257,
            model_name="gpt2-medium/dropout",
            hf_model="gpt2-medium",
        )
    elif name == "gpt2-medium/lora":
        return TrainingConfig(
            n_layers=24,
            n_heads=16,
            embedding_dim=1024,
            dropout_rate=0,
            use_bias=True,
            block_size=1024,
            vocab_size=50257,
            lora_rank=1,
            model_name="gpt2-medium/lora",
            hf_model="gpt2-medium",
        )
    elif name == 'gpt2-large':
        return TrainingConfig(
            n_layers=36,
            n_heads=20,
            embedding_dim=1280,
            dropout_rate=0,
            use_bias=True,
            block_size=1024,
            vocab_size=50257,
            model_name="gpt2-large",
            hf_model="gpt2-large",
        )
    elif name == 'gpt2-large/lora':
        return TrainingConfig(
            n_layers=36,
            n_heads=20,
            embedding_dim=1280,
            dropout_rate=0,
            use_bias=True,
            block_size=1024,
            vocab_size=50257,
            lora_rank=1,
            model_name="gpt2-large/lora",
            hf_model="gpt2-large",
        )
    elif name == "gpt2-xl":
        return TrainingConfig(
            n_layers=48,
            n_heads=25,
            embedding_dim=1600,
            dropout_rate=0,
            use_bias=True,
            block_size=1024,
            vocab_size=50257,
            model_name="gpt2-xl",
            hf_model="gpt2-xl",
        )
    elif name == "gpt2-xl/dropout":
        return TrainingConfig(n_layers=48,
                              n_heads=25,
                              embedding_dim=1600,
                              dropout_rate=0.2,
                              use_bias=True,
                              block_size=1024,
                              vocab_size=50257,
                              model_name="gpt2-xl/dropout",
                              hf_model="gpt2-xl")
    elif name == "gpt2-xl/lora":
        return TrainingConfig(
            n_layers=48,
            n_heads=25,
            embedding_dim=1600,
            dropout_rate=0,
            use_bias=True,
            block_size=1024,
            vocab_size=50257,
            lora_rank=1,
            model_name="gpt2-xl/lora",
            hf_model="gpt2-xl",
        )
