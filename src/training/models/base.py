"""Abstract base classes for MT models."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Union
from pathlib import Path


class AbstractMTModel(ABC):
    """
    Abstract base class for all machine translation models.

    Provides a unified interface for translation with extensibility.
    """

    def __init__(self, model_config: Any):
        """
        Initialize MT model.

        Args:
            model_config: Model configuration object
        """
        self.config = model_config
        self.model = None
        self.tokenizer = None
        self._loaded = False

    @abstractmethod
    def load(self):
        """Load model and tokenizer."""
        pass

    @abstractmethod
    def translate(
        self,
        texts: Union[str, List[str]],
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Translate text(s).

        Args:
            texts: Single text or list of texts
            source_lang: Source language code (overrides config)
            target_lang: Target language code (overrides config)
            **kwargs: Additional generation parameters

        Returns:
            Translated text(s)
        """
        pass

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata."""
        return {
            "model_name": getattr(self.config, "model_name_or_path", "unknown"),
            "model_type": getattr(self.config, "model_type", "unknown"),
            "architecture": getattr(self.config, "architecture", "unknown"),
            "loaded": self._loaded,
        }

    def __repr__(self) -> str:
        info = self.get_model_info()
        return f"{self.__class__.__name__}({info['model_name']}, loaded={info['loaded']})"


class AbstractTrainableMTModel(AbstractMTModel):
    """
    Abstract base class for trainable MT models.

    Extends AbstractMTModel with training capabilities.
    """

    @abstractmethod
    def prepare_for_training(
        self,
        lora_config: Optional[Any] = None,
        training_config: Optional[Any] = None
    ):
        """
        Prepare model for training (e.g., apply LoRA, set trainable params).

        Args:
            lora_config: LoRA configuration
            training_config: Training configuration
        """
        pass

    @abstractmethod
    def save(self, output_dir: str, save_full_model: bool = False):
        """
        Save model (full model or adapters only).

        Args:
            output_dir: Output directory
            save_full_model: Save full model or just adapters
        """
        pass

    @abstractmethod
    def load_adapter(self, adapter_path: str):
        """
        Load LoRA adapter weights.

        Args:
            adapter_path: Path to adapter weights
        """
        pass

    def get_trainable_parameters(self) -> Dict[str, int]:
        """
        Get information about trainable parameters.

        Returns:
            Dict with total, trainable, and trainable percentage
        """
        if self.model is None:
            return {"total": 0, "trainable": 0, "percentage": 0.0}

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            "total": total_params,
            "trainable": trainable_params,
            "percentage": 100 * trainable_params / total_params if total_params > 0 else 0.0,
        }

    def print_trainable_parameters(self):
        """Print trainable parameters information."""
        params_info = self.get_trainable_parameters()
        print(f"Total parameters: {params_info['total']:,}")
        print(f"Trainable parameters: {params_info['trainable']:,}")
        print(f"Trainable %: {params_info['percentage']:.2f}%")

    def freeze_base_model(self):
        """Freeze all base model parameters (for LoRA)."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")

        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_all_parameters(self):
        """Unfreeze all parameters (for full fine-tuning)."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load() first.")

        for param in self.model.parameters():
            param.requires_grad = True
