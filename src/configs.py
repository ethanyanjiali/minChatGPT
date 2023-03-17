from dataclasses import dataclass, asdict


@dataclass
class TrainingConfig:
    # Transformer
    n_layers: int
    n_heads: int
    embedding_dim: int
    dropout_rate: float
    use_bias: bool
    block_size: int
    vocab_size: int
    model_name: str
    hf_model: str
    grad_clip: float = 1.0
    exp_name: str = ""
    batch_size: int = 1
    lr: float = 0.0001
    lora_rank: int = 0
    pretrain: str = "huggingface"
    activation_checkpointing: bool = False
    finetune_method: str = ""
    total_epochs: int = 1
    # SFT specific
    max_steps: int = 20000
    # PPO specific
    actor_weights: str = ""
    critic_weights: str = ""
    reward_model_weights: str = ""
    sft_model_weights: str = ""
    actor_lr: float = 5e-6
    critic_lr: float = 9e-6
    kl_beta: float = 0.02
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95

    def dict(self):
        return {k: str(v) for k, v in asdict(self).items()}


def get_configs(name) -> TrainingConfig:
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
            finetune_method="lora",
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
    elif name == 'gpt2-large/dropout':
        return TrainingConfig(
            n_layers=36,
            n_heads=20,
            embedding_dim=1280,
            dropout_rate=0,
            use_bias=True,
            block_size=1024,
            vocab_size=50257,
            model_name="gpt2-large/dropout",
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
            finetune_method="lora",
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
            finetune_method="lora",
        )
