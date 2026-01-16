"""Integration strategies for linguistic features in MT models."""

from src.training.linguistic.integration.encoder_augmentation import (
    EncoderAugmentation,
    augment_encoder_inputs,
)

__all__ = [
    "EncoderAugmentation",
    "augment_encoder_inputs",
]
