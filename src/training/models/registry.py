"""Model registry for extensible model registration."""

from typing import Dict, Type, Optional, Any, Callable
from ..models.base import AbstractMTModel


class ModelRegistry:
    """
    Registry for MT models with decorator-based registration.

    Enables easy extension with custom models without modifying core code.

    Example:
        @ModelRegistry.register("my_custom_model")
        class MyCustomModel(AbstractTrainableMTModel):
            def load(self): ...
            def translate(self, texts): ...
    """

    _registry: Dict[str, Type[AbstractMTModel]] = {}
    _model_info: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        model_type: Optional[str] = None,
        architecture: Optional[str] = None,
        description: Optional[str] = None
    ) -> Callable:
        """
        Decorator for registering a model class.

        Args:
            name: Unique identifier for the model
            model_type: Type of model (seq2seq, causal_lm, encoder_only)
            architecture: Model architecture (nllb, llama, etc.)
            description: Human-readable description

        Returns:
            Decorator function

        Example:
            @ModelRegistry.register("nllb-600m", model_type="seq2seq", architecture="nllb")
            class NLLBModel(AbstractTrainableMTModel):
                ...
        """
        def decorator(model_class: Type[AbstractMTModel]) -> Type[AbstractMTModel]:
            if not issubclass(model_class, AbstractMTModel):
                raise TypeError(f"{model_class.__name__} must inherit from AbstractMTModel")

            if name in cls._registry:
                raise ValueError(f"Model '{name}' is already registered")

            cls._registry[name] = model_class
            cls._model_info[name] = {
                "class": model_class.__name__,
                "model_type": model_type,
                "architecture": architecture,
                "description": description or f"{model_class.__name__} model",
            }

            return model_class

        return decorator

    @classmethod
    def get(cls, name: str) -> Type[AbstractMTModel]:
        """
        Get model class by name.

        Args:
            name: Registered model name

        Returns:
            Model class

        Raises:
            ValueError: If model not found
        """
        if name not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Model '{name}' not found in registry. "
                f"Available models: {available}"
            )

        return cls._registry[name]

    @classmethod
    def create(cls, name: str, model_config: Any, **kwargs) -> AbstractMTModel:
        """
        Create model instance by name.

        Args:
            name: Registered model name
            model_config: Model configuration
            **kwargs: Additional arguments for model initialization

        Returns:
            Model instance
        """
        model_class = cls.get(name)
        return model_class(model_config, **kwargs)

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if model is registered."""
        return name in cls._registry

    @classmethod
    def list_models(cls) -> Dict[str, Dict[str, Any]]:
        """
        List all registered models with metadata.

        Returns:
            Dictionary of model names to metadata
        """
        return cls._model_info.copy()

    @classmethod
    def unregister(cls, name: str):
        """
        Unregister a model (for testing or dynamic removal).

        Args:
            name: Model name to unregister
        """
        if name in cls._registry:
            del cls._registry[name]
            del cls._model_info[name]

    @classmethod
    def clear(cls):
        """Clear all registered models (for testing)."""
        cls._registry.clear()
        cls._model_info.clear()

    @classmethod
    def register_from_dict(cls, config: Dict[str, Any]):
        """
        Register multiple models from configuration dictionary.

        Args:
            config: Dictionary mapping model names to class references

        Example:
            ModelRegistry.register_from_dict({
                "nllb-600m": {"class": NLLBModel, "model_type": "seq2seq"},
                "llama-8b": {"class": LlamaModel, "model_type": "causal_lm"},
            })
        """
        for name, info in config.items():
            model_class = info.get("class")
            if model_class is None:
                raise ValueError(f"Missing 'class' for model '{name}'")

            cls.register(
                name=name,
                model_type=info.get("model_type"),
                architecture=info.get("architecture"),
                description=info.get("description")
            )(model_class)

    @classmethod
    def print_registered_models(cls):
        """Print all registered models in a formatted table."""
        if not cls._registry:
            print("No models registered.")
            return

        print("\n" + "=" * 80)
        print("REGISTERED MODELS")
        print("=" * 80)

        for name, info in cls._model_info.items():
            print(f"\nName: {name}")
            print(f"  Class: {info['class']}")
            print(f"  Type: {info['model_type'] or 'N/A'}")
            print(f"  Architecture: {info['architecture'] or 'N/A'}")
            print(f"  Description: {info['description']}")

        print("=" * 80 + "\n")

    @classmethod
    def get_models_by_type(cls, model_type: str) -> Dict[str, Type[AbstractMTModel]]:
        """
        Get all models of a specific type.

        Args:
            model_type: Model type to filter by (seq2seq, causal_lm, etc.)

        Returns:
            Dictionary of matching models
        """
        return {
            name: cls._registry[name]
            for name, info in cls._model_info.items()
            if info["model_type"] == model_type
        }

    @classmethod
    def get_models_by_architecture(cls, architecture: str) -> Dict[str, Type[AbstractMTModel]]:
        """
        Get all models of a specific architecture.

        Args:
            architecture: Architecture to filter by (nllb, llama, etc.)

        Returns:
            Dictionary of matching models
        """
        return {
            name: cls._registry[name]
            for name, info in cls._model_info.items()
            if info["architecture"] == architecture
        }
