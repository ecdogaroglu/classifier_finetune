# Description: Configuration for the model training.

from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration for model training."""
    model_name: str = "distilbert/distilroberta-base"
    max_length: int = 128
    batch_size: int = 16
    num_epochs: int = 5
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    min_samples_per_class: int = 100
    max_samples_per_class: int = 1000
    frozen_layers: int = 10
    max_grad_norm: float = None
