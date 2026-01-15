"""Highly configurable training configuration."""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from pathlib import Path
import yaml


@dataclass
class TrainingConfig:
    """
    Comprehensive training configuration with full extensibility.

    All parameters are configurable and can be loaded from YAML or dict.
    New parameters can be added without breaking existing code.
    """

    # Output configuration
    output_dir: str = "models/trained"
    run_name: Optional[str] = None
    logging_dir: Optional[str] = "logs/training"

    # Training hyperparameters
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Optimizer configuration
    optim: str = "adamw_torch"  # adamw_torch, adamw_hf, adafactor, etc.
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8

    # Learning rate schedule
    lr_scheduler_type: str = "linear"  # linear, cosine, polynomial, etc.
    warmup_steps: int = 300
    warmup_ratio: float = 0.0

    # Evaluation configuration
    evaluation_strategy: str = "steps"  # no, steps, epoch
    eval_steps: Optional[int] = 500
    eval_accumulation_steps: Optional[int] = None

    # Saving configuration
    save_strategy: str = "steps"  # no, steps, epoch
    save_steps: int = 1000
    save_total_limit: Optional[int] = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.0

    # Logging
    logging_steps: int = 100
    logging_first_step: bool = True
    logging_strategy: str = "steps"
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])

    # Mixed precision training
    fp16: bool = False
    bf16: bool = False
    fp16_opt_level: str = "O1"
    fp16_backend: str = "auto"

    # Memory optimization
    gradient_checkpointing: bool = False
    optim_target_modules: Optional[List[str]] = None

    # Data configuration
    max_source_length: int = 128
    max_target_length: int = 128
    preprocessing_num_workers: Optional[int] = None
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = True

    # Generation configuration (for evaluation)
    num_beams: int = 4
    max_new_tokens: int = 128
    do_sample: bool = False
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0

    # Reproducibility
    seed: int = 42
    data_seed: Optional[int] = None

    # Device configuration
    device: str = "cuda"
    local_rank: int = -1
    ddp_find_unused_parameters: Optional[bool] = None

    # Resume training
    resume_from_checkpoint: Optional[str] = None
    ignore_data_skip: bool = False

    # Advanced options
    label_smoothing_factor: float = 0.0
    predict_with_generate: bool = True
    include_inputs_for_metrics: bool = False

    # Custom parameters (extensible)
    custom_params: Dict[str, Any] = field(default_factory=dict)

    # V100 GPU optimizations
    auto_find_batch_size: bool = False
    v100_optimized: bool = False  # Auto-applies V100-specific settings

    def __post_init__(self):
        """Apply V100 optimizations if enabled."""
        if self.v100_optimized:
            self.per_device_train_batch_size = 24
            self.gradient_accumulation_steps = 2
            self.fp16 = True
            self.dataloader_num_workers = 4
            self.dataloader_pin_memory = True
            self.gradient_checkpointing = True

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TrainingConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrainingConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)

    def to_yaml(self, yaml_path: str):
        """Save configuration to YAML file."""
        Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    def update(self, **kwargs):
        """Update configuration with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                # Store unknown parameters in custom_params
                self.custom_params[key] = value

    def merge(self, other: "TrainingConfig") -> "TrainingConfig":
        """Merge with another configuration (other takes precedence)."""
        merged_dict = {**self.to_dict(), **other.to_dict()}
        return TrainingConfig.from_dict(merged_dict)
