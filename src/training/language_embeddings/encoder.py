"""
Language embedding encoder.

Encodes language embeddings to the desired dimension and prepares them
for integration with neural MT models.
"""

from typing import List, Optional, Dict
import torch
import torch.nn as nn
import numpy as np
import logging

from src.training.language_embeddings.loaders.base import (
    LanguageEmbedding,
    AbstractLanguageEmbeddingLoader,
)

logger = logging.getLogger(__name__)


class LanguageEmbeddingEncoder(nn.Module):
    """
    Encoder for language embeddings.

    Projects language embeddings to desired dimension and optionally
    applies transformations.
    """

    def __init__(
        self,
        loader: AbstractLanguageEmbeddingLoader,
        output_dim: int,
        use_projection: bool = True,
        dropout: float = 0.1,
    ):
        """
        Initialize language embedding encoder.

        Args:
            loader: Language embedding loader
            output_dim: Output dimension for embeddings
            use_projection: Whether to use learned projection (vs fixed)
            dropout: Dropout rate
        """
        super().__init__()
        self.loader = loader
        self.output_dim = output_dim
        self.use_projection = use_projection

        # Get input dimension from loader
        self.input_dim = loader.get_embedding_dimension()

        # Create projection layer
        if use_projection and self.input_dim != output_dim:
            self.projection = nn.Sequential(
                nn.Linear(self.input_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.Dropout(dropout),
            )
        else:
            self.projection = None

        # Cache for encoded embeddings
        self._cache: Dict[str, torch.Tensor] = {}

        logger.info(
            f"LanguageEmbeddingEncoder initialized: "
            f"input_dim={self.input_dim}, output_dim={output_dim}"
        )

    def encode_language(
        self,
        language_code: str,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Encode language embedding.

        Args:
            language_code: ISO 639-3 language code
            device: Device to place tensor on

        Returns:
            Encoded language embedding tensor [output_dim]
        """
        # Check cache
        cache_key = f"{language_code}_{device}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Load raw embedding
        lang_emb = self.loader.load_embedding(language_code)

        # Convert to tensor
        embedding_tensor = torch.from_numpy(lang_emb.embedding).float()

        # Project if needed
        if self.projection is not None:
            with torch.no_grad():
                embedding_tensor = self.projection(embedding_tensor)
        elif self.input_dim != self.output_dim:
            # Pad or truncate
            if self.input_dim < self.output_dim:
                # Pad with zeros
                padding = torch.zeros(self.output_dim - self.input_dim)
                embedding_tensor = torch.cat([embedding_tensor, padding])
            else:
                # Truncate
                embedding_tensor = embedding_tensor[:self.output_dim]

        # Move to device
        if device is not None:
            embedding_tensor = embedding_tensor.to(device)

        # Cache
        self._cache[cache_key] = embedding_tensor

        return embedding_tensor

    def encode_batch(
        self,
        language_codes: List[str],
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Encode batch of language embeddings.

        Args:
            language_codes: List of ISO 639-3 language codes
            device: Device to place tensor on

        Returns:
            Batch of encoded embeddings [batch_size, output_dim]
        """
        embeddings = []
        for code in language_codes:
            emb = self.encode_language(code, device)
            embeddings.append(emb)

        return torch.stack(embeddings, dim=0)

    def encode_source_target_pair(
        self,
        source_lang: str,
        target_lang: str,
        device: Optional[torch.device] = None,
        concatenate: bool = True,
    ) -> torch.Tensor:
        """
        Encode source-target language pair.

        Args:
            source_lang: Source language code
            target_lang: Target language code
            device: Device to place tensor on
            concatenate: If True, concatenate; else subtract

        Returns:
            Encoded language pair [output_dim] or [2*output_dim]
        """
        src_emb = self.encode_language(source_lang, device)
        tgt_emb = self.encode_language(target_lang, device)

        if concatenate:
            return torch.cat([src_emb, tgt_emb], dim=0)
        else:
            # Language difference vector (direction of translation)
            return tgt_emb - src_emb

    def clear_cache(self):
        """Clear embedding cache."""
        self._cache.clear()

    def forward(
        self,
        language_codes: List[str],
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Forward pass (for consistency with nn.Module).

        Args:
            language_codes: List of language codes
            device: Device to place tensor on

        Returns:
            Batch of encoded embeddings
        """
        return self.encode_batch(language_codes, device)


def create_language_encoder(
    embedding_source: str,
    output_dim: int,
    feature_types: Optional[List[str]] = None,
    use_projection: bool = True,
    **loader_kwargs
) -> LanguageEmbeddingEncoder:
    """
    Create language embedding encoder.

    Args:
        embedding_source: Source name ("uriel", "wals", "lang2vec")
        output_dim: Output dimension
        feature_types: Feature types to use
        use_projection: Whether to use learned projection
        **loader_kwargs: Additional arguments for loader

    Returns:
        LanguageEmbeddingEncoder instance
    """
    from src.training.language_embeddings.loaders.base import LoaderRegistry

    # Create loader
    loader = LoaderRegistry.create_loader(
        embedding_source,
        feature_types=feature_types,
        **loader_kwargs
    )

    # Create encoder
    encoder = LanguageEmbeddingEncoder(
        loader=loader,
        output_dim=output_dim,
        use_projection=use_projection,
    )

    return encoder


# Example usage
def example_usage():
    """Example of using language embedding encoder."""
    # Create encoder
    encoder = create_language_encoder(
        embedding_source="uriel",
        output_dim=64,
        feature_types=["syntax", "phonology"],
    )

    # Encode single language
    hi_emb = encoder.encode_language("hi")
    print(f"Hindi embedding shape: {hi_emb.shape}")

    # Encode language pair
    pair_emb = encoder.encode_source_target_pair("hi", "bho", concatenate=True)
    print(f"Hindi-Bhojpuri pair embedding shape: {pair_emb.shape}")

    # Encode batch
    batch_emb = encoder.encode_batch(["hi", "en", "bho"])
    print(f"Batch embedding shape: {batch_emb.shape}")
