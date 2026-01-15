"""Model factory for creating models from configurations."""

from typing import Optional, Dict, Any, Union
from pathlib import Path
import yaml

from .registry import ModelRegistry
from .base import AbstractMTModel
from ..config import ModelConfig


class ModelFactory:
    """
    Factory for creating MT models from configurations.

    Supports creation from:
    - ModelConfig objects
    - YAML files
    - Python dictionaries
    - Model registry names
    """

    @staticmethod
    def create_model(
        config: Union[ModelConfig, str, Dict[str, Any]],
        auto_load: bool = True,
        **kwargs
    ) -> AbstractMTModel:
        """
        Create model from configuration.

        Args:
            config: Model configuration (ModelConfig, YAML path, or dict)
            auto_load: Automatically load model weights
            **kwargs: Additional arguments passed to model constructor

        Returns:
            Model instance

        Example:
            # From ModelConfig
            config = ModelConfig.create_preset("nllb-600m", "hi", "bho")
            model = ModelFactory.create_model(config)

            # From YAML
            model = ModelFactory.create_model("config/models/nllb_600m.yaml")

            # From dict
            model = ModelFactory.create_model({
                "model_name_or_path": "facebook/nllb-200-distilled-600M",
                "source_lang": "hi",
                "target_lang": "bho"
            })
        """
        # Convert to ModelConfig if needed
        if isinstance(config, str):
            if Path(config).exists():
                # Load from YAML file
                model_config = ModelConfig.from_yaml(config)
            else:
                # Try as preset name
                raise ValueError(f"Configuration file not found: {config}")
        elif isinstance(config, dict):
            model_config = ModelConfig.from_dict(config)
        elif isinstance(config, ModelConfig):
            model_config = config
        else:
            raise TypeError(f"Invalid config type: {type(config)}")

        # Determine which model class to use
        model_class = ModelFactory._get_model_class(model_config)

        # Create model instance
        model = model_class(model_config, **kwargs)

        # Load model weights if requested
        if auto_load:
            model.load()

        return model

    @staticmethod
    def _get_model_class(config: ModelConfig):
        """
        Determine model class from configuration.

        Args:
            config: Model configuration

        Returns:
            Model class
        """
        # Check if registry name is specified
        if config.registry_name is not None:
            if ModelRegistry.is_registered(config.registry_name):
                return ModelRegistry.get(config.registry_name)
            else:
                raise ValueError(f"Model '{config.registry_name}' not found in registry")

        # Auto-detect based on architecture
        if config.architecture is not None:
            # Try to find registered model by architecture
            arch_models = ModelRegistry.get_models_by_architecture(config.architecture)
            if arch_models:
                # Return first match (can be made more sophisticated)
                return list(arch_models.values())[0]

        # Fallback to generic HuggingFace model based on model_type
        if config.model_type == "seq2seq":
            # Import here to avoid circular dependency
            from .hf_seq2seq import HuggingFaceSeq2SeqModel
            return HuggingFaceSeq2SeqModel
        elif config.model_type == "causal_lm":
            from .hf_llm import HuggingFaceLLMModel
            return HuggingFaceLLMModel
        else:
            raise ValueError(
                f"Cannot determine model class for model_type='{config.model_type}'. "
                "Please specify registry_name in ModelConfig or register a model."
            )

    @staticmethod
    def create_from_preset(
        preset_name: str,
        source_lang: str,
        target_lang: str,
        auto_load: bool = True,
        **kwargs
    ) -> AbstractMTModel:
        """
        Create model from preset name.

        Args:
            preset_name: Preset name (e.g., "nllb-600m", "llama3-8b")
            source_lang: Source language code
            target_lang: Target language code
            auto_load: Automatically load model weights
            **kwargs: Additional arguments

        Returns:
            Model instance

        Example:
            model = ModelFactory.create_from_preset("nllb-600m", "hi", "bho")
        """
        config = ModelConfig.create_preset(preset_name, source_lang, target_lang)
        return ModelFactory.create_model(config, auto_load=auto_load, **kwargs)

    @staticmethod
    def create_from_registry(
        registry_name: str,
        model_config: ModelConfig,
        auto_load: bool = True,
        **kwargs
    ) -> AbstractMTModel:
        """
        Create model from registry by name.

        Args:
            registry_name: Name in ModelRegistry
            model_config: Model configuration
            auto_load: Automatically load model weights
            **kwargs: Additional arguments

        Returns:
            Model instance

        Example:
            config = ModelConfig(model_name_or_path="facebook/nllb-200-distilled-600M")
            model = ModelFactory.create_from_registry("nllb", config)
        """
        model_class = ModelRegistry.get(registry_name)
        model = model_class(model_config, **kwargs)

        if auto_load:
            model.load()

        return model

    @staticmethod
    def list_available_presets() -> Dict[str, str]:
        """
        List available preset configurations.

        Returns:
            Dictionary of preset names to descriptions
        """
        presets = {
            "nllb-600m": "NLLB 600M distilled model (16GB GPU)",
            "nllb-1.3b": "NLLB 1.3B model (32GB GPU)",
            "nllb-3.3b": "NLLB 3.3B model (40GB+ GPU)",
            "mbart-large-50": "mBART Large 50-language model (16GB GPU)",
            "mt5-small": "mT5 Small model (8GB GPU)",
            "mt5-base": "mT5 Base model (16GB GPU)",
            "indictrans2-200m": "IndicTrans2 200M distilled (8GB GPU)",
            "indictrans2-1b": "IndicTrans2 1B (16GB GPU)",
            "llama3-8b": "Llama 3 8B Instruct (16GB GPU with 8-bit)",
            "mistral-7b": "Mistral 7B Instruct (16GB GPU with 8-bit)",
        }
        return presets

    @staticmethod
    def print_available_presets():
        """Print available preset configurations."""
        presets = ModelFactory.list_available_presets()

        print("\n" + "=" * 80)
        print("AVAILABLE MODEL PRESETS")
        print("=" * 80)

        for name, description in presets.items():
            print(f"\n{name}")
            print(f"  {description}")

        print("\n" + "=" * 80)
        print("Usage: ModelFactory.create_from_preset('preset_name', 'src_lang', 'tgt_lang')")
        print("=" * 80 + "\n")
