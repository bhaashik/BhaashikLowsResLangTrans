"""Configuration management for the translation system."""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from dotenv import load_dotenv


class Config:
    """Configuration manager that loads settings from YAML and environment variables."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.

        Args:
            config_path: Path to config YAML file. If None, uses default location.
        """
        # Load environment variables
        load_dotenv()

        # Determine config path
        if config_path is None:
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "config.yaml"

        # Load YAML configuration
        with open(config_path, "r") as f:
            self._config = yaml.safe_load(f)

        # Set up base directory
        self.base_dir = Path(os.getenv("BASE_DIR", "/mnt/data/nlp_resources"))

        # Set up HuggingFace configuration
        self.hf_token = os.getenv("HF_TOKEN")
        self.hf_datasets_cache = os.getenv("HF_DATASETS_CACHE", str(self.base_dir / "cache" / "datasets"))
        self.hf_home = os.getenv("HF_HOME", str(self.base_dir / "cache" / "huggingface"))
        self.transformers_cache = os.getenv("TRANSFORMERS_CACHE", str(self.base_dir / "cache" / "transformers"))

        # API Keys
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.google_credentials = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        self.azure_key = os.getenv("AZURE_TRANSLATOR_KEY")
        self.azure_region = os.getenv("AZURE_TRANSLATOR_REGION")

        # Processing configuration
        self.batch_size = int(os.getenv("DEFAULT_BATCH_SIZE", self._config.get("processing", {}).get("batch_size", 32)))
        self.max_length = int(os.getenv("MAX_LENGTH", self._config.get("processing", {}).get("max_length", 512)))
        self.num_beams = int(os.getenv("NUM_BEAMS", self._config.get("processing", {}).get("num_beams", 5)))

        # Logging
        self.log_level = os.getenv("LOG_LEVEL", self._config.get("logging", {}).get("level", "INFO"))
        self.log_file = os.getenv("LOG_FILE", str(self.base_dir / "logs" / "translation.log"))

        # Cost tracking
        self.enable_cost_tracking = os.getenv("ENABLE_COST_TRACKING", "true").lower() == "true"
        self.cost_log_file = os.getenv("COST_LOG_FILE", str(self.base_dir / "logs" / "cost_tracking.json"))

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.

        Args:
            key: Configuration key (e.g., 'models.indictrans2.default')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    def get_supported_languages(self) -> list:
        """Get list of IndicTrans2 supported languages."""
        return self.get("languages.indictrans2_supported", [])

    def get_unsupported_languages(self) -> Dict[str, Dict]:
        """Get dictionary of unsupported languages with metadata."""
        return self.get("languages.unsupported", {})

    def get_model_path(self, model_type: str, variant: str = "default") -> str:
        """
        Get model path from configuration.

        Args:
            model_type: Type of model (e.g., 'indictrans2', 'nllb', 'llms')
            variant: Model variant (e.g., 'en_to_indic_1b', 'default')

        Returns:
            HuggingFace model path
        """
        return self.get(f"models.{model_type}.{variant}", "")

    def get_translation_strategy(self, language_code: str) -> Dict:
        """
        Get translation strategy for a given language.

        Args:
            language_code: Language code (e.g., 'hi', 'bho')

        Returns:
            Dictionary with strategy details
        """
        supported = self.get_supported_languages()

        if language_code in supported:
            return {
                "method": "indictrans2_direct",
                "cost": 0,
                "description": "Direct translation using IndicTrans2"
            }
        else:
            return self.get("strategy.unsupported_strategy", {})

    def get_api_costs(self, provider: str, model: str) -> Dict[str, float]:
        """
        Get API costs for a specific provider and model.

        Args:
            provider: API provider (e.g., 'anthropic', 'openai')
            model: Model name (e.g., 'haiku_4_5', 'gpt_4_turbo')

        Returns:
            Dictionary with cost information
        """
        return self.get(f"api.{provider}.models.{model}", {})

    def get_paths(self) -> Dict[str, Path]:
        """
        Get all configured paths.

        Returns:
            Dictionary of path names to Path objects
        """
        paths_config = self.get("paths", {})
        return {
            name: self.base_dir / path
            for name, path in paths_config.items()
        }

    def ensure_directories(self):
        """Create all necessary directories if they don't exist."""
        paths = self.get_paths()
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)

        # Also create cache directories
        Path(self.hf_datasets_cache).mkdir(parents=True, exist_ok=True)
        Path(self.hf_home).mkdir(parents=True, exist_ok=True)
        Path(self.transformers_cache).mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        return f"Config(base_dir={self.base_dir})"
