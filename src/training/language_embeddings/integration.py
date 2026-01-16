"""
Integration strategies for language embeddings in MT models.

Provides different methods to integrate language embeddings:
1. Concatenation: Concatenate with encoder/decoder representations
2. Addition: Add to encoder/decoder representations
3. Conditioning: Use as conditioning vector (via attention)
4. Adapter: Use adapter layers conditioned on language embedding
"""

from typing import Optional
import torch
import torch.nn as nn
import logging

from src.training.language_embeddings.encoder import LanguageEmbeddingEncoder

logger = logging.getLogger(__name__)


class LanguageAwareEncoder(nn.Module):
    """
    Encoder wrapper that integrates language embeddings.

    Adds language-specific information to encoder representations.
    """

    def __init__(
        self,
        base_encoder: nn.Module,
        language_encoder: LanguageEmbeddingEncoder,
        source_lang: str,
        target_lang: Optional[str] = None,
        integration_method: str = "concatenate",
        use_target_embedding: bool = False,
    ):
        """
        Initialize language-aware encoder.

        Args:
            base_encoder: Base encoder module
            language_encoder: Language embedding encoder
            source_lang: Source language code
            target_lang: Target language code (optional)
            integration_method: Integration method ("concatenate", "add", "condition")
            use_target_embedding: Whether to use target embedding
        """
        super().__init__()
        self.base_encoder = base_encoder
        self.language_encoder = language_encoder
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.integration_method = integration_method
        self.use_target_embedding = use_target_embedding

        # Get dimensions
        try:
            self.hidden_dim = base_encoder.config.d_model
        except AttributeError:
            self.hidden_dim = base_encoder.config.hidden_size

        self.lang_emb_dim = language_encoder.output_dim

        # Create integration layers
        if integration_method == "concatenate":
            # Project concatenated representation back to original dimension
            if use_target_embedding:
                input_dim = self.hidden_dim + 2 * self.lang_emb_dim
            else:
                input_dim = self.hidden_dim + self.lang_emb_dim
            self.projection = nn.Linear(input_dim, self.hidden_dim)
        elif integration_method == "add":
            # Project language embedding to hidden dimension
            self.projection = nn.Linear(self.lang_emb_dim, self.hidden_dim)
        elif integration_method == "condition":
            # Cross-attention layer for conditioning
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=self.hidden_dim,
                num_heads=8,
                dropout=0.1,
                batch_first=True,
            )
        else:
            raise ValueError(f"Unknown integration method: {integration_method}")

        logger.info(
            f"LanguageAwareEncoder initialized: "
            f"method={integration_method}, "
            f"use_target={use_target_embedding}"
        )

    def forward(
        self,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        Forward pass with language embedding integration.

        Args:
            inputs_embeds: Input embeddings [batch, seq_len, hidden_dim]
            attention_mask: Attention mask [batch, seq_len]
            **kwargs: Additional arguments for base encoder

        Returns:
            Encoder outputs with language information integrated
        """
        # Run base encoder
        encoder_outputs = self.base_encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs
        )

        # Get hidden states
        hidden_states = encoder_outputs.last_hidden_state  # [batch, seq_len, hidden]

        # Get language embedding
        device = hidden_states.device
        batch_size = hidden_states.shape[0]

        if self.use_target_embedding and self.target_lang:
            # Use source-target pair embedding
            lang_emb = self.language_encoder.encode_source_target_pair(
                self.source_lang,
                self.target_lang,
                device=device,
                concatenate=True  # [2 * lang_emb_dim]
            )
        else:
            # Use source embedding only
            lang_emb = self.language_encoder.encode_language(
                self.source_lang,
                device=device
            )

        # Expand to batch
        lang_emb = lang_emb.unsqueeze(0).expand(batch_size, -1)  # [batch, lang_dim]

        # Integrate language embedding
        if self.integration_method == "concatenate":
            # Expand to sequence length
            lang_emb_expanded = lang_emb.unsqueeze(1).expand(
                -1, hidden_states.shape[1], -1
            )  # [batch, seq_len, lang_dim]

            # Concatenate and project
            concatenated = torch.cat([hidden_states, lang_emb_expanded], dim=2)
            integrated_states = self.projection(concatenated)

        elif self.integration_method == "add":
            # Project language embedding and add
            lang_emb_projected = self.projection(lang_emb)  # [batch, hidden_dim]
            lang_emb_expanded = lang_emb_projected.unsqueeze(1)  # [batch, 1, hidden]
            integrated_states = hidden_states + lang_emb_expanded

        elif self.integration_method == "condition":
            # Use cross-attention to condition on language embedding
            lang_emb_expanded = lang_emb.unsqueeze(1)  # [batch, 1, lang_dim]

            # Cross-attention: query=hidden_states, key=value=lang_emb
            conditioned_states, _ = self.cross_attention(
                query=hidden_states,
                key=lang_emb_expanded,
                value=lang_emb_expanded,
                need_weights=False,
            )
            integrated_states = hidden_states + conditioned_states

        # Update outputs
        encoder_outputs.last_hidden_state = integrated_states

        return encoder_outputs


def add_language_embeddings(
    model: nn.Module,
    language_encoder: LanguageEmbeddingEncoder,
    source_lang: str,
    target_lang: Optional[str] = None,
    integration_method: str = "concatenate",
    use_target_embedding: bool = False,
) -> nn.Module:
    """
    Add language embeddings to an existing model.

    Args:
        model: Base seq2seq model
        language_encoder: Language embedding encoder
        source_lang: Source language code
        target_lang: Target language code
        integration_method: Integration method
        use_target_embedding: Whether to use target embedding

    Returns:
        Model with language embeddings integrated
    """
    logger.info("Adding language embeddings to model...")

    # Get base encoder
    try:
        base_encoder = model.get_encoder()
    except AttributeError:
        raise ValueError("Model must have get_encoder() method")

    # Create language-aware encoder
    lang_aware_encoder = LanguageAwareEncoder(
        base_encoder=base_encoder,
        language_encoder=language_encoder,
        source_lang=source_lang,
        target_lang=target_lang,
        integration_method=integration_method,
        use_target_embedding=use_target_embedding,
    )

    # Replace encoder
    # Note: This is a simplified version. In practice, we'd need to handle
    # model-specific encoder replacement more carefully.
    if hasattr(model, 'encoder'):
        model.encoder = lang_aware_encoder
    elif hasattr(model, 'model') and hasattr(model.model, 'encoder'):
        model.model.encoder = lang_aware_encoder
    else:
        logger.warning("Could not replace encoder. Model structure not recognized.")

    logger.info("✓ Language embeddings integrated")

    return model


class LanguageEmbeddingAdapter(nn.Module):
    """
    Adapter layer conditioned on language embedding.

    Adds language-specific transformation via adapter layers.
    """

    def __init__(
        self,
        hidden_dim: int,
        language_emb_dim: int,
        adapter_dim: int = 64,
    ):
        """
        Initialize language embedding adapter.

        Args:
            hidden_dim: Hidden dimension of model
            language_emb_dim: Language embedding dimension
            adapter_dim: Adapter bottleneck dimension
        """
        super().__init__()

        # Down-projection conditioned on language
        self.down_project = nn.Linear(hidden_dim + language_emb_dim, adapter_dim)

        # Up-projection back to hidden dimension
        self.up_project = nn.Linear(adapter_dim, hidden_dim)

        self.activation = nn.ReLU()

    def forward(
        self,
        hidden_states: torch.Tensor,
        language_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through adapter.

        Args:
            hidden_states: Hidden states [batch, seq_len, hidden_dim]
            language_embedding: Language embedding [batch, lang_dim]

        Returns:
            Adapted hidden states [batch, seq_len, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Expand language embedding to sequence length
        lang_emb_expanded = language_embedding.unsqueeze(1).expand(-1, seq_len, -1)

        # Concatenate with hidden states
        concatenated = torch.cat([hidden_states, lang_emb_expanded], dim=-1)

        # Down-project
        down = self.down_project(concatenated)
        down = self.activation(down)

        # Up-project
        up = self.up_project(down)

        # Residual connection
        output = hidden_states + up

        return output


# Example usage
def example_usage():
    """Example of integrating language embeddings."""
    from transformers import AutoModelForSeq2SeqLM
    from src.training.language_embeddings.encoder import create_language_encoder

    # Create language encoder
    lang_encoder = create_language_encoder(
        embedding_source="uriel",
        output_dim=64,
        feature_types=["syntax", "phonology"],
    )

    # Load base model
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "facebook/nllb-200-distilled-600M"
    )

    # Add language embeddings
    model = add_language_embeddings(
        model=model,
        language_encoder=lang_encoder,
        source_lang="hi",
        target_lang="bho",
        integration_method="concatenate",
        use_target_embedding=True,
    )

    print("Model with language embeddings ready for training!")
