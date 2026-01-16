"""
Fine-tuned model provider for locally trained translation models.

This provider integrates locally fine-tuned models (NLLB, mBART, mT5, IndicTrans2,
Llama, Mistral) with the universal_translate framework, allowing them to be used
seamlessly alongside API-based providers.

Supports:
- LoRA adapters and merged models
- All models from src/training/models (seq2seq and causal LMs)
- Batch translation
- No API costs (local inference)
- GPU/CPU inference
"""

import os
from typing import Optional, Dict, Any, List
from pathlib import Path
import logging

from ..core import (
    BaseTranslator,
    TranslationRequest,
    TranslationResponse,
    TranslationResult,
    TranslationStatus,
    CostEstimate,
    TranslationError,
)

# Import training infrastructure
from src.training.models import ModelFactory, ModelRegistry
from src.training.config import ModelConfig, LoRAConfig

logger = logging.getLogger(__name__)


class FineTunedProvider(BaseTranslator):
    """
    Translation provider for locally fine-tuned models.

    This provider enables:
    - Using fine-tuned models from src/training/ in the universal pipeline
    - Loading LoRA adapters or merged models
    - Zero-cost local inference
    - Batch processing for efficiency

    Example:
        provider = FineTunedProvider(
            model_path="models/trained/nllb-bhojpuri-lora",
            base_model="nllb-600m",
            adapter_only=True
        )
        request = TranslationRequest(
            units=["नमस्ते"],
            source_lang="hi",
            target_lang="bho"
        )
        response = provider.translate_sync(request)
    """

    def __init__(
        self,
        model_path: str,
        base_model: Optional[str] = None,
        adapter_only: bool = False,
        device: str = "cuda",
        batch_size: int = 16,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize fine-tuned model provider.

        Args:
            model_path: Path to fine-tuned model or adapter
            base_model: Base model name (required if adapter_only=True)
            adapter_only: Whether model_path contains only LoRA adapter
            device: Device to run inference on ("cuda" or "cpu")
            batch_size: Batch size for translation
            config: Additional configuration
        """
        super().__init__(
            name="finetuned",
            model=model_path,
            config=config or {}
        )

        self.model_path = Path(model_path)
        self.base_model = base_model
        self.adapter_only = adapter_only
        self.device = device
        self.batch_size = batch_size

        self.model = None
        self._model_type = None

        # Validate configuration
        if adapter_only and not base_model:
            raise ValueError("base_model required when adapter_only=True")

        logger.info(
            f"Initialized FineTunedProvider: model_path={model_path}, "
            f"adapter_only={adapter_only}, device={device}"
        )

    def initialize(self):
        """Load the fine-tuned model."""
        if self._initialized:
            return

        logger.info(f"Loading fine-tuned model from {self.model_path}")

        try:
            if self.adapter_only:
                # Load base model with adapter
                self._load_with_adapter()
            else:
                # Load merged/full model
                self._load_full_model()

            self._initialized = True
            logger.info("Fine-tuned model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load fine-tuned model: {e}")
            raise TranslationError(
                f"Failed to load model from {self.model_path}: {e}",
                provider="finetuned",
                details={"model_path": str(self.model_path)}
            )

    def _load_with_adapter(self):
        """Load base model and apply LoRA adapter."""
        logger.info(f"Loading base model {self.base_model} with adapter")

        # Create base model config
        model_config = ModelConfig.create_preset(
            self.base_model,
            source_lang="auto",  # Will be set during translation
            target_lang="auto"
        )
        model_config.device = self.device

        # Create model using factory
        self.model = ModelFactory.create_from_config(model_config)

        # Load adapter
        adapter_path = self.model_path
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter path not found: {adapter_path}")

        logger.info(f"Loading adapter from {adapter_path}")
        self.model.load_adapter(str(adapter_path))

        self._model_type = model_config.model_type

    def _load_full_model(self):
        """Load merged/full fine-tuned model."""
        logger.info(f"Loading full model from {self.model_path}")

        # Detect model type from path or use base_model hint
        if self.base_model:
            model_config = ModelConfig.create_preset(
                self.base_model,
                source_lang="auto",
                target_lang="auto"
            )
        else:
            # Create generic config
            model_config = ModelConfig(
                model_name_or_path=str(self.model_path),
                model_type="seq2seq",  # Default, will auto-detect
                source_lang="auto",
                target_lang="auto",
            )

        model_config.device = self.device

        # Override model path to load from fine-tuned location
        model_config.model_name_or_path = str(self.model_path)

        # Create model
        self.model = ModelFactory.create_from_config(model_config)
        self._model_type = model_config.model_type

    async def translate(self, request: TranslationRequest) -> TranslationResponse:
        """
        Translate text units asynchronously.

        Args:
            request: Translation request

        Returns:
            Translation response

        Note: This provider runs synchronously but provides async interface
        for compatibility.
        """
        return self.translate_sync(request)

    def translate_sync(self, request: TranslationRequest) -> TranslationResponse:
        """
        Translate text units synchronously.

        Args:
            request: Translation request with units, languages

        Returns:
            Translation response with results
        """
        if not self._initialized:
            self.initialize()

        logger.info(
            f"Translating {len(request.units)} units from "
            f"{request.source_lang} to {request.target_lang}"
        )

        try:
            # Translate in batches for efficiency
            translations = []
            for i in range(0, len(request.units), self.batch_size):
                batch = request.units[i:i + self.batch_size]

                batch_translations = self.model.translate(
                    texts=batch,
                    source_lang=request.source_lang,
                    target_lang=request.target_lang,
                )

                translations.extend(batch_translations)

            # Create translation results
            results = [
                TranslationResult(
                    unit=unit,
                    translation=translation,
                    status=TranslationStatus.SUCCESS
                )
                for unit, translation in zip(request.units, translations)
            ]

            # Create response
            response = TranslationResponse(
                results=results,
                status=TranslationStatus.SUCCESS,
                metadata={
                    "provider": "finetuned",
                    "model": str(self.model_path),
                    "model_type": self._model_type,
                    "adapter_only": self.adapter_only,
                    "device": self.device,
                    "batch_size": self.batch_size,
                }
            )

            # Add cost info (zero for local models)
            response.cost_info = CostEstimate(
                estimated_cost=0.0,
                currency="USD",
                breakdown={
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "input_cost": 0.0,
                    "output_cost": 0.0,
                    "total_cost": 0.0,
                }
            )

            logger.info(
                f"Translation completed: {len(translations)} units, "
                f"cost=$0 (local inference)"
            )

            return response

        except Exception as e:
            logger.error(f"Translation failed: {e}")
            raise TranslationError(
                f"Translation failed: {e}",
                provider="finetuned",
                details={
                    "model_path": str(self.model_path),
                    "num_units": len(request.units),
                }
            )

    def get_cost_estimate(self, request: TranslationRequest) -> CostEstimate:
        """
        Estimate cost (always zero for local models).

        Args:
            request: Translation request

        Returns:
            Cost estimate (zero)
        """
        return CostEstimate(
            estimated_cost=0.0,
            currency="USD",
            breakdown={
                "input_tokens": 0,
                "output_tokens": 0,
                "input_cost": 0.0,
                "output_cost": 0.0,
                "total_cost": 0.0,
                "note": "Local inference - no API costs"
            }
        )

    def supports_batch(self) -> bool:
        """Fine-tuned models support batch translation."""
        return True

    def supports_prompt_caching(self) -> bool:
        """Local models don't use prompt caching (not applicable)."""
        return False

    def get_max_tokens(self) -> int:
        """Get maximum tokens for the model."""
        # Default limits based on model type
        if self._model_type == "seq2seq":
            return 512  # Typical for NLLB/mBART/mT5
        else:
            return 2048  # For causal LMs

    def get_rate_limits(self) -> Dict[str, int]:
        """
        Get rate limits (none for local models).

        Returns:
            Empty dict (no rate limits)
        """
        return {
            "requests_per_minute": -1,  # Unlimited
            "tokens_per_minute": -1,  # Unlimited
            "note": "Local inference has no rate limits"
        }

    def cleanup(self):
        """Unload model and free memory."""
        if self.model is not None:
            logger.info("Unloading fine-tuned model")
            del self.model
            self.model = None

            # Force garbage collection
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self._initialized = False


def create_from_training_run(
    training_output_dir: str,
    adapter_only: bool = True,
    **kwargs
) -> FineTunedProvider:
    """
    Create provider from training run output directory.

    Args:
        training_output_dir: Output directory from LoRATrainer
        adapter_only: Whether to load only adapter (default: True)
        **kwargs: Additional arguments for FineTunedProvider

    Returns:
        FineTunedProvider instance
    """
    training_dir = Path(training_output_dir)

    # Find adapter or model
    if adapter_only:
        # Look for adapter_model.safetensors or adapter_model.bin
        adapter_files = list(training_dir.glob("adapter_model.*"))
        if not adapter_files:
            raise FileNotFoundError(
                f"No adapter found in {training_dir}. "
                "Expected adapter_model.safetensors or adapter_model.bin"
            )
        model_path = str(training_dir)
    else:
        # Use merged model directory
        merged_dir = training_dir / "merged"
        if not merged_dir.exists():
            raise FileNotFoundError(
                f"Merged model not found at {merged_dir}. "
                "Run merge step first or use adapter_only=True"
            )
        model_path = str(merged_dir)

    # Try to detect base model from config
    config_file = training_dir / "config.json"
    base_model = None
    if config_file.exists():
        import json
        with open(config_file) as f:
            config = json.load(f)
            base_model = config.get("_name_or_path")

    return FineTunedProvider(
        model_path=model_path,
        base_model=base_model,
        adapter_only=adapter_only,
        **kwargs
    )
