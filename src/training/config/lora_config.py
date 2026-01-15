"""LoRA (Low-Rank Adaptation) configuration."""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
import yaml
from pathlib import Path


@dataclass
class LoRAConfig:
    """
    Configurable LoRA parameters for parameter-efficient fine-tuning.

    Supports all PEFT library LoRA parameters and is fully extensible.
    """

    # Core LoRA parameters
    r: int = 16  # Rank of the low-rank matrices
    lora_alpha: int = 32  # Scaling parameter (usually 2*r)
    lora_dropout: float = 0.1  # Dropout probability

    # Target modules (model-specific)
    target_modules: Optional[List[str]] = None  # Auto-detected if None

    # Predefined target module sets for common architectures
    target_modules_seq2seq: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "out_proj",
        "fc1", "fc2"
    ])
    target_modules_llm: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

    # LoRA variant
    use_rslora: bool = False  # Rank-stabilized LoRA
    use_dora: bool = False  # Weight-Decomposed Low-Rank Adaptation

    # Bias handling
    bias: str = "none"  # none, all, lora_only

    # Task type
    task_type: str = "SEQ_2_SEQ_LM"  # SEQ_2_SEQ_LM, CAUSAL_LM

    # Module selection
    modules_to_save: Optional[List[str]] = None  # Additional modules to train

    # Initialization
    lora_init_type: str = "default"  # default, gaussian, xavier

    # Layer selection
    layers_to_transform: Optional[List[int]] = None  # Specific layers (None = all)
    layers_pattern: Optional[str] = None  # Regex pattern for layer selection

    # Advanced options
    fan_in_fan_out: bool = False
    enable_lora: Optional[List[bool]] = None

    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Auto-configure target modules based on task type if not specified."""
        if self.target_modules is None:
            if self.task_type == "SEQ_2_SEQ_LM":
                self.target_modules = self.target_modules_seq2seq
            elif self.task_type == "CAUSAL_LM":
                self.target_modules = self.target_modules_llm

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "LoRAConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "LoRAConfig":
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

    def to_peft_config(self):
        """Convert to PEFT LoraConfig object."""
        try:
            from peft import LoraConfig as PEFTLoraConfig
        except ImportError:
            raise ImportError("PEFT library required. Install: pip install peft")

        return PEFTLoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules,
            lora_dropout=self.lora_dropout,
            bias=self.bias,
            task_type=self.task_type,
            modules_to_save=self.modules_to_save,
            use_rslora=self.use_rslora,
            use_dora=self.use_dora,
            fan_in_fan_out=self.fan_in_fan_out,
            init_lora_weights=self.lora_init_type,
            layers_to_transform=self.layers_to_transform,
            layers_pattern=self.layers_pattern,
        )

    @classmethod
    def create_preset(cls, preset: str = "default") -> "LoRAConfig":
        """
        Create LoRA configuration from preset.

        Presets:
        - default: Standard LoRA (r=16, alpha=32)
        - efficient: Memory-efficient (r=8, alpha=16)
        - high_rank: Higher capacity (r=32, alpha=64)
        - seq2seq: Optimized for seq2seq models
        - llm: Optimized for causal LLMs
        """
        presets = {
            "default": {"r": 16, "lora_alpha": 32},
            "efficient": {"r": 8, "lora_alpha": 16, "lora_dropout": 0.05},
            "high_rank": {"r": 32, "lora_alpha": 64},
            "seq2seq": {
                "r": 16,
                "lora_alpha": 32,
                "task_type": "SEQ_2_SEQ_LM",
                "target_modules": ["q_proj", "v_proj", "k_proj", "out_proj"]
            },
            "llm": {
                "r": 16,
                "lora_alpha": 32,
                "task_type": "CAUSAL_LM",
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"]
            },
        }

        if preset not in presets:
            raise ValueError(f"Unknown preset: {preset}. Choose from {list(presets.keys())}")

        return cls(**presets[preset])
