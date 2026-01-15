"""Data configuration for training."""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import yaml


@dataclass
class DataConfig:
    """
    Comprehensive data configuration for parallel corpus loading and preprocessing.

    Fully configurable and extensible for various data formats and pipelines.
    """

    # Data paths
    data_path: str = "data/parallel/corpus.tsv"
    train_file: Optional[str] = None  # Separate train file
    eval_file: Optional[str] = None  # Separate eval file
    test_file: Optional[str] = None  # Separate test file

    # Data format
    data_format: str = "tsv"  # tsv, csv, json, jsonl, huggingface
    dataset_name: Optional[str] = None  # HuggingFace dataset name
    dataset_config: Optional[str] = None  # HuggingFace dataset config

    # Column names (for TSV/CSV/JSON)
    source_column: str = "source"
    target_column: str = "target"
    delimiter: str = "\t"  # For TSV/CSV

    # Language configuration
    source_lang: str = "hi"
    target_lang: str = "bho"

    # Data splitting
    split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)  # train, eval, test
    shuffle_before_split: bool = True
    stratify_by: Optional[str] = None  # Column to stratify by

    # Data filtering
    min_source_length: int = 1
    max_source_length: int = 512
    min_target_length: int = 1
    max_target_length: int = 512
    min_length_ratio: float = 0.5  # target/source
    max_length_ratio: float = 2.0
    remove_empty: bool = True
    remove_duplicates: bool = False

    # Preprocessing
    preprocessing_type: str = "none"  # none, indic_trans, spacy, custom
    lowercase: bool = False
    normalize_punctuation: bool = False
    remove_extra_whitespace: bool = True

    # Data augmentation
    augmentation_enabled: bool = False
    augmentation_methods: List[str] = field(default_factory=list)  # backtranslation, paraphrase, noise
    augmentation_ratio: float = 0.0  # 0.0 = no augmentation, 0.5 = 50% augmented data

    # Backtranslation augmentation
    backtranslation_model: Optional[str] = None
    backtranslation_num_samples: int = 1

    # Paraphrase augmentation
    paraphrase_model: Optional[str] = None
    paraphrase_temperature: float = 0.7

    # Noise augmentation
    noise_types: List[str] = field(default_factory=list)  # char_swap, word_drop, word_shuffle
    noise_probability: float = 0.1

    # Caching
    cache_dir: Optional[str] = "data/cache"
    overwrite_cache: bool = False

    # Sampling
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    max_test_samples: Optional[int] = None

    # Quality filtering
    quality_filtering_enabled: bool = False
    min_quality_score: float = 0.5
    quality_metric: str = "length_ratio"  # length_ratio, alignment_score, custom

    # Corpus analysis
    analyze_corpus: bool = False
    analysis_output_dir: Optional[str] = "data/analysis"

    # Custom preprocessing function
    custom_preprocessor: Optional[str] = None  # Path to custom preprocessor module

    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and auto-configure data settings."""
        # Validate split ratios
        if abs(sum(self.split_ratios) - 1.0) > 0.01:
            raise ValueError(f"Split ratios must sum to 1.0, got {sum(self.split_ratios)}")

        # Auto-detect format if not specified
        if self.data_format == "tsv":
            self.delimiter = "\t"
        elif self.data_format == "csv":
            self.delimiter = ","

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "DataConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "DataConfig":
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
    def create_preset(cls, preset: str, data_path: str) -> "DataConfig":
        """
        Create data configuration from preset.

        Presets:
        - default: Standard TSV format with basic filtering
        - strict: Aggressive filtering for high-quality data
        - augmented: Includes data augmentation
        - large_scale: Optimized for large datasets
        """
        presets = {
            "default": {
                "data_format": "tsv",
                "remove_empty": True,
                "remove_duplicates": False,
            },
            "strict": {
                "data_format": "tsv",
                "remove_empty": True,
                "remove_duplicates": True,
                "min_source_length": 5,
                "min_target_length": 5,
                "min_length_ratio": 0.6,
                "max_length_ratio": 1.5,
                "quality_filtering_enabled": True,
                "min_quality_score": 0.7,
            },
            "augmented": {
                "data_format": "tsv",
                "augmentation_enabled": True,
                "augmentation_methods": ["backtranslation"],
                "augmentation_ratio": 0.3,
            },
            "large_scale": {
                "data_format": "huggingface",
                "cache_dir": "data/cache",
                "overwrite_cache": False,
                "shuffle_before_split": True,
            },
        }

        if preset not in presets:
            raise ValueError(f"Unknown preset: {preset}. Choose from {list(presets.keys())}")

        config = presets[preset].copy()
        config["data_path"] = data_path

        return cls(**config)

    def get_file_paths(self) -> Dict[str, Optional[str]]:
        """Get all data file paths."""
        return {
            "train": self.train_file or self.data_path,
            "eval": self.eval_file,
            "test": self.test_file,
        }

    def should_split_data(self) -> bool:
        """Check if data needs to be split."""
        return self.train_file is None or self.eval_file is None
