"""
Extended model wrapper with linguistic features support.

This module provides a wrapper around HuggingFace models that integrates
linguistic features (dependency parses) into the translation process.
"""

from typing import List, Optional, Union, Dict, Any
import torch
import torch.nn as nn
from transformers import PreTrainedModel
import logging

from src.training.models.base import AbstractTrainableMTModel
from src.training.config import ModelConfig, LoRAConfig, TrainingConfig
from src.training.linguistic import (
    LinguisticFeaturesConfig,
    EncoderAugmentation,
)

logger = logging.getLogger(__name__)


class LinguisticEnhancedModel(nn.Module):
    """
    Wrapper that adds linguistic features to a seq2seq model.

    This wraps the base HuggingFace model and integrates parse features
    into the encoder through concatenation with word embeddings.
    """

    def __init__(
        self,
        base_model: PreTrainedModel,
        linguistic_config: LinguisticFeaturesConfig,
        source_lang: str,
        target_lang: Optional[str] = None,
    ):
        """
        Initialize linguistic-enhanced model.

        Args:
            base_model: Base HuggingFace model
            linguistic_config: Linguistic features configuration
            source_lang: Source language code
            target_lang: Target language code
        """
        super().__init__()
        self.base_model = base_model
        self.linguistic_config = linguistic_config
        self.source_lang = source_lang
        self.target_lang = target_lang

        # Initialize encoder augmentation
        if linguistic_config.use_source_parse or linguistic_config.use_target_parse:
            self.augmentation = EncoderAugmentation(
                linguistic_config,
                source_lang,
                target_lang
            )
        else:
            self.augmentation = None

        # Get embedding dimension from base model
        self.embed_dim = base_model.config.d_model

        # Create projection layer to handle concatenated features
        # New input dim = embed_dim + linguistic_encoding_dim
        self.input_projection = None
        if self.augmentation:
            augmented_dim = self.embed_dim + linguistic_config.encoding_dim
            self.input_projection = nn.Linear(augmented_dim, self.embed_dim)

        logger.info(
            f"LinguisticEnhancedModel initialized: "
            f"embed_dim={self.embed_dim}, "
            f"linguistic_dim={linguistic_config.encoding_dim if self.augmentation else 0}"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        source_texts: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Forward pass with linguistic feature integration.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            decoder_input_ids: Decoder input IDs [batch, tgt_len]
            decoder_attention_mask: Decoder attention mask [batch, tgt_len]
            labels: Labels for training [batch, tgt_len]
            source_texts: Original source texts (for parsing)
            **kwargs: Additional arguments

        Returns:
            Model outputs
        """
        # If no linguistic features or no source texts, use base model
        if self.augmentation is None or source_texts is None:
            return self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                labels=labels,
                **kwargs
            )

        # Get embeddings from base model
        encoder = self.base_model.get_encoder()
        embed_tokens = encoder.embed_tokens

        # Get word embeddings [batch, seq_len, embed_dim]
        word_embeddings = embed_tokens(input_ids)

        # Augment with linguistic features
        try:
            augmented_embeddings = self.augmentation.augment_source(
                source_texts,
                word_embeddings,
                use_cache=True
            )

            # Project back to original dimension
            if self.input_projection:
                projected_embeddings = self.input_projection(augmented_embeddings)
            else:
                projected_embeddings = augmented_embeddings

        except Exception as e:
            logger.warning(f"Failed to augment with linguistic features: {e}")
            # Fallback to original embeddings
            projected_embeddings = word_embeddings

        # Run encoder with augmented embeddings
        encoder_outputs = encoder(
            inputs_embeds=projected_embeddings,
            attention_mask=attention_mask,
            **{k: v for k, v in kwargs.items() if k.startswith('encoder_')}
        )

        # Run decoder
        decoder_outputs = self.base_model(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            **{k: v for k, v in kwargs.items() if not k.startswith('encoder_')}
        )

        return decoder_outputs

    def generate(self, *args, **kwargs):
        """
        Generate translations.

        For generation, we use the base model directly since we don't have
        access to source texts in generate().
        """
        return self.base_model.generate(*args, **kwargs)

    def cleanup(self):
        """Cleanup linguistic resources."""
        if self.augmentation:
            self.augmentation.cleanup()


class LinguisticModelWrapper:
    """
    Wrapper to make LinguisticEnhancedModel compatible with trainers.

    This allows trainers to pass source_texts to the model during training.
    """

    def __init__(
        self,
        base_model_wrapper: AbstractTrainableMTModel,
        linguistic_config: Optional[LinguisticFeaturesConfig] = None,
    ):
        """
        Initialize wrapper.

        Args:
            base_model_wrapper: Base model wrapper (HuggingFaceSeq2SeqModel, etc.)
            linguistic_config: Linguistic features configuration
        """
        self.base_wrapper = base_model_wrapper
        self.linguistic_config = linguistic_config
        self.linguistic_model = None

    def enable_linguistic_features(self):
        """Enable linguistic features by wrapping the model."""
        if self.linguistic_config is None:
            logger.warning("No linguistic config provided, features not enabled")
            return

        if not self.base_wrapper._loaded:
            raise RuntimeError("Base model not loaded. Call load() first.")

        logger.info("Enabling linguistic features...")

        # Create linguistic-enhanced model
        self.linguistic_model = LinguisticEnhancedModel(
            base_model=self.base_wrapper.model,
            linguistic_config=self.linguistic_config,
            source_lang=self.base_wrapper.config.source_lang,
            target_lang=self.base_wrapper.config.target_lang,
        )

        # Replace base model with linguistic model
        self.base_wrapper.model = self.linguistic_model

        logger.info("✓ Linguistic features enabled")

    def prepare_for_training(
        self,
        lora_config: LoRAConfig,
        training_config: TrainingConfig
    ):
        """Prepare model for training (applies LoRA, etc.)."""
        # Enable linguistic features before training
        if self.linguistic_config:
            self.enable_linguistic_features()

        # Prepare base model (applies LoRA to base_model inside linguistic_model)
        self.base_wrapper.prepare_for_training(lora_config, training_config)

    def __getattr__(self, name):
        """Delegate attribute access to base wrapper."""
        return getattr(self.base_wrapper, name)


def create_linguistic_model(
    base_model_class: type,
    model_config: ModelConfig,
    linguistic_config: Optional[LinguisticFeaturesConfig] = None,
) -> Union[AbstractTrainableMTModel, LinguisticModelWrapper]:
    """
    Create model with optional linguistic features.

    Args:
        base_model_class: Base model class (e.g., HuggingFaceSeq2SeqModel)
        model_config: Model configuration
        linguistic_config: Linguistic features configuration

    Returns:
        Model wrapper (with or without linguistic features)
    """
    # Create base model
    base_model = base_model_class(model_config)

    # Wrap with linguistic features if config provided
    if linguistic_config and (
        linguistic_config.use_source_parse or linguistic_config.use_target_parse
    ):
        return LinguisticModelWrapper(base_model, linguistic_config)

    return base_model


# Example usage
def example_usage():
    """Example of creating and using linguistic-enhanced model."""
    from src.training.models import HuggingFaceSeq2SeqModel
    from src.training.config import ModelConfig
    from src.training.linguistic import LinguisticFeaturesConfig

    # Model configuration
    model_config = ModelConfig(
        model_name_or_path="facebook/nllb-200-distilled-600M",
        source_lang="hi",
        target_lang="bho",
    )

    # Linguistic features configuration
    linguistic_config = LinguisticFeaturesConfig(
        use_source_parse=True,
        parser="stanza",
        features=["dependency_labels", "pos_tags"],
        integration_method="encoder_augmentation",
        encoding_dim=128,
    )

    # Create model with linguistic features
    model = create_linguistic_model(
        HuggingFaceSeq2SeqModel,
        model_config,
        linguistic_config
    )

    # Load model
    model.load()

    # Model is now ready for training with linguistic features
    # The trainer will automatically pass source_texts to the model
