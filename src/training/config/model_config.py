"""Model configuration for training."""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from pathlib import Path
import yaml


@dataclass
class ModelConfig:
    """
    Comprehensive model configuration supporting all model types.

    Extensible to support new models without code changes.
    """

    # Model identification
    model_name_or_path: str = "facebook/nllb-200-distilled-600M"
    model_type: str = "seq2seq"  # seq2seq, causal_lm, encoder_only
    model_class: Optional[str] = None  # Custom model class name

    # Model architecture
    architecture: Optional[str] = None  # nllb, mbart, mt5, llama, mistral, etc.

    # Tokenizer configuration
    tokenizer_name_or_path: Optional[str] = None  # Defaults to model_name_or_path
    use_fast_tokenizer: bool = True
    tokenizer_revision: Optional[str] = None

    # Language configuration
    source_lang: str = "hi"
    target_lang: str = "bho"
    source_lang_code: Optional[str] = None  # Model-specific language code
    target_lang_code: Optional[str] = None

    # Loading options
    cache_dir: Optional[str] = None
    model_revision: Optional[str] = "main"
    use_auth_token: Optional[str] = None
    trust_remote_code: bool = False

    # Quantization
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_use_double_quant: bool = False
    bnb_4bit_quant_type: str = "nf4"

    # Device mapping
    device_map: Optional[str] = None  # auto, balanced, sequential, or custom
    max_memory: Optional[Dict[int, str]] = None  # e.g., {0: "20GB", 1: "20GB"}
    offload_folder: Optional[str] = None

    # Model parameters
    torch_dtype: Optional[str] = None  # auto, float32, float16, bfloat16
    low_cpu_mem_usage: bool = True

    # LoRA configuration (if using LoRA)
    use_lora: bool = False
    lora_config: Optional[Dict[str, Any]] = None

    # Generation configuration
    forced_bos_token_id: Optional[int] = None
    forced_eos_token_id: Optional[int] = None
    max_length: int = 256
    min_length: int = 1
    num_beams: int = 4
    length_penalty: float = 1.0

    # Model-specific parameters
    model_params: Dict[str, Any] = field(default_factory=dict)

    # Custom model registry name (for extension)
    registry_name: Optional[str] = None

    # Preprocessing options
    add_prefix: bool = False  # For models like mT5
    prefix_template: Optional[str] = None  # "translate {source} to {target}: "

    def __post_init__(self):
        """Auto-configure model-specific settings."""
        # Set tokenizer path if not specified
        if self.tokenizer_name_or_path is None:
            self.tokenizer_name_or_path = self.model_name_or_path

        # Auto-detect architecture from model name
        if self.architecture is None:
            self.architecture = self._detect_architecture()

        # Set model type based on architecture
        if self.model_type == "seq2seq" and self.architecture is None:
            if "nllb" in self.model_name_or_path.lower():
                self.architecture = "nllb"
            elif "mbart" in self.model_name_or_path.lower():
                self.architecture = "mbart"
            elif "mt5" in self.model_name_or_path.lower():
                self.architecture = "mt5"

    def _detect_architecture(self) -> Optional[str]:
        """Auto-detect model architecture from name."""
        name_lower = self.model_name_or_path.lower()

        architectures = {
            "nllb": ["nllb"],
            "mbart": ["mbart"],
            "mt5": ["mt5"],
            "indictrans2": ["indictrans2"],
            "llama": ["llama"],
            "mistral": ["mistral"],
            "mixtral": ["mixtral"],
        }

        for arch, keywords in architectures.items():
            if any(kw in name_lower for kw in keywords):
                return arch

        return None

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ModelConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ModelConfig":
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

    @classmethod
    def create_preset(cls, model_name: str, source_lang: str, target_lang: str) -> "ModelConfig":
        """
        Create model configuration from preset.

        Supported models:
        - nllb-600m, nllb-1.3b, nllb-3.3b
        - mbart-large-50
        - mt5-small, mt5-base
        - indictrans2-200m, indictrans2-1b
        - llama3-8b
        - mistral-7b
        """
        presets = {
            "nllb-600m": {
                "model_name_or_path": "facebook/nllb-200-distilled-600M",
                "model_type": "seq2seq",
                "architecture": "nllb",
            },
            "nllb-1.3b": {
                "model_name_or_path": "facebook/nllb-200-1.3B",
                "model_type": "seq2seq",
                "architecture": "nllb",
            },
            "nllb-3.3b": {
                "model_name_or_path": "facebook/nllb-200-3.3B",
                "model_type": "seq2seq",
                "architecture": "nllb",
            },
            "mbart-large-50": {
                "model_name_or_path": "facebook/mbart-large-50-many-to-many-mmt",
                "model_type": "seq2seq",
                "architecture": "mbart",
            },
            "mt5-small": {
                "model_name_or_path": "google/mt5-small",
                "model_type": "seq2seq",
                "architecture": "mt5",
                "add_prefix": True,
            },
            "mt5-base": {
                "model_name_or_path": "google/mt5-base",
                "model_type": "seq2seq",
                "architecture": "mt5",
                "add_prefix": True,
            },
            "indictrans2-200m": {
                "model_name_or_path": "ai4bharat/indictrans2-en-indic-dist-200M",
                "model_type": "seq2seq",
                "architecture": "indictrans2",
            },
            "indictrans2-1b": {
                "model_name_or_path": "ai4bharat/indictrans2-en-indic-1B",
                "model_type": "seq2seq",
                "architecture": "indictrans2",
            },
            "llama3-8b": {
                "model_name_or_path": "meta-llama/Meta-Llama-3-8B-Instruct",
                "model_type": "causal_lm",
                "architecture": "llama",
                "load_in_8bit": True,
            },
            "mistral-7b": {
                "model_name_or_path": "mistralai/Mistral-7B-Instruct-v0.2",
                "model_type": "causal_lm",
                "architecture": "mistral",
                "load_in_8bit": True,
            },
        }

        if model_name not in presets:
            raise ValueError(f"Unknown model: {model_name}. Choose from {list(presets.keys())}")

        config = presets[model_name].copy()
        config.update({
            "source_lang": source_lang,
            "target_lang": target_lang,
        })

        return cls(**config)
