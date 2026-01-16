"""
Language-aware model wrapper.

Integrates language embeddings into seq2seq models for improved translation.
Can be combined with linguistic features for maximum benefit.
"""

from typing import Optional
import torch
import torch.nn as nn
from transformers import PreTrainedModel
import logging

from src.training.models.base import AbstractTrainableMTModel
from src.training.config import ModelConfig, LoRAConfig, TrainingConfig
from src.training.language_embeddings import (
    LanguageEmbeddingsConfig,
    LanguageEmbeddingEncoder,
    create_language_encoder,
)

logger = logging.getLogger(__name__)


class LanguageAwareModel(nn.Module):
    """
    Wrapper that adds language embeddings to a seq2seq model.

    Integrates typological features of source/target languages into
    the model to improve translation quality.
    """

    def __init__(
        self,
        base_model: PreTrainedModel,
        language_embeddings_config: LanguageEmbeddingsConfig,
        source_lang: str,
        target_lang: str,
    ):
        """
        Initialize language-aware model.

        Args:
            base_model: Base HuggingFace model
            language_embeddings_config: Language embeddings configuration
            source_lang: Source language code
            target_lang: Target language code
        """
        super().__init__()
        self.base_model = base_model
        self.config_le = language_embeddings_config
        self.source_lang = source_lang
        self.target_lang = target_lang

        # Get model dimensions
        self.hidden_dim = base_model.config.d_model
        self.lang_emb_dim = language_embeddings_config.embedding_dim

        # Create language embedding encoder
        self.language_encoder = create_language_encoder(
            embedding_source=language_embeddings_config.embedding_source,
            output_dim=language_embeddings_config.embedding_dim,
            feature_types=language_embeddings_config.feature_types,
            use_projection=True,
        )

        # Create integration layer based on method
        self._create_integration_layers()

        logger.info(
            f"LanguageAwareModel initialized: "
            f"source={language_embeddings_config.embedding_source}, "
            f"method={language_embeddings_config.integration_method}, "
            f"dim={language_embeddings_config.embedding_dim}"
        )

    def _create_integration_layers(self):
        """Create integration layers based on configuration."""
        method = self.config_le.integration_method

        if method == "concatenate":
            # Project concatenated [hidden; lang_emb] back to hidden_dim
            if self.config_le.use_both:
                # Source + target embeddings
                input_dim = self.hidden_dim + 2 * self.lang_emb_dim
            else:
                # Single embedding
                input_dim = self.hidden_dim + self.lang_emb_dim

            self.integration_projection = nn.Linear(input_dim, self.hidden_dim)

        elif method == "add":
            # Project language embedding to hidden_dim for addition
            self.integration_projection = nn.Linear(
                self.lang_emb_dim,
                self.hidden_dim
            )

        elif method == "condition":
            # Cross-attention for conditioning
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=self.hidden_dim,
                num_heads=8,
                dropout=0.1,
                batch_first=True,
            )

        elif method == "adapter":
            # Adapter layers
            adapter_dim = self.config_le.custom_params.get("adapter_dim", 64)
            self.adapter_down = nn.Linear(
                self.hidden_dim + self.lang_emb_dim,
                adapter_dim
            )
            self.adapter_up = nn.Linear(adapter_dim, self.hidden_dim)
            self.adapter_activation = nn.ReLU()

        else:
            raise ValueError(f"Unknown integration method: {method}")

    def _get_language_embedding(self, device: torch.device) -> torch.Tensor:
        """
        Get language embedding for current language pair.

        Args:
            device: Device to place tensor on

        Returns:
            Language embedding tensor
        """
        if self.config_le.use_both:
            # Concatenate source and target embeddings
            lang_emb = self.language_encoder.encode_source_target_pair(
                self.source_lang,
                self.target_lang,
                device=device,
                concatenate=True,
            )
        elif self.config_le.use_source_embedding:
            lang_emb = self.language_encoder.encode_language(
                self.source_lang,
                device=device
            )
        else:  # use_target_embedding
            lang_emb = self.language_encoder.encode_language(
                self.target_lang,
                device=device
            )

        return lang_emb

    def _integrate_language_embedding(
        self,
        hidden_states: torch.Tensor,
        language_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Integrate language embedding with hidden states.

        Args:
            hidden_states: Hidden states [batch, seq_len, hidden_dim]
            language_embedding: Language embedding [lang_dim]

        Returns:
            Integrated hidden states [batch, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device

        # Expand language embedding to batch
        lang_emb_batch = language_embedding.unsqueeze(0).expand(batch_size, -1)

        method = self.config_le.integration_method

        if method == "concatenate":
            # Expand to sequence length and concatenate
            lang_emb_expanded = lang_emb_batch.unsqueeze(1).expand(-1, seq_len, -1)
            concatenated = torch.cat([hidden_states, lang_emb_expanded], dim=2)
            integrated = self.integration_projection(concatenated)

        elif method == "add":
            # Project and add
            lang_emb_projected = self.integration_projection(lang_emb_batch)
            lang_emb_expanded = lang_emb_projected.unsqueeze(1)
            integrated = hidden_states + lang_emb_expanded

        elif method == "condition":
            # Cross-attention conditioning
            lang_emb_expanded = lang_emb_batch.unsqueeze(1)  # [batch, 1, lang_dim]
            conditioned, _ = self.cross_attention(
                query=hidden_states,
                key=lang_emb_expanded,
                value=lang_emb_expanded,
                need_weights=False,
            )
            integrated = hidden_states + conditioned

        elif method == "adapter":
            # Adapter transformation
            lang_emb_expanded = lang_emb_batch.unsqueeze(1).expand(-1, seq_len, -1)
            concatenated = torch.cat([hidden_states, lang_emb_expanded], dim=-1)
            adapted = self.adapter_down(concatenated)
            adapted = self.adapter_activation(adapted)
            adapted = self.adapter_up(adapted)
            integrated = hidden_states + adapted

        return integrated

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        Forward pass with language embedding integration.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            decoder_input_ids: Decoder input IDs [batch, tgt_len]
            decoder_attention_mask: Decoder attention mask [batch, tgt_len]
            labels: Labels for training [batch, tgt_len]
            **kwargs: Additional arguments

        Returns:
            Model outputs
        """
        # Get language embedding
        lang_emb = self._get_language_embedding(device=input_ids.device)

        # Get encoder
        encoder = self.base_model.get_encoder()
        embed_tokens = encoder.embed_tokens

        # Get word embeddings [batch, seq_len, hidden_dim]
        word_embeddings = embed_tokens(input_ids)

        # Integrate language embedding with word embeddings
        integrated_embeddings = self._integrate_language_embedding(
            word_embeddings,
            lang_emb
        )

        # Run encoder with integrated embeddings
        encoder_outputs = encoder(
            inputs_embeds=integrated_embeddings,
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

        For generation, we need to handle language embeddings carefully.
        """
        # For now, use base model generate
        # TODO: Properly integrate language embeddings during generation
        return self.base_model.generate(*args, **kwargs)

    def cleanup(self):
        """Cleanup resources."""
        if hasattr(self, 'language_encoder'):
            self.language_encoder.clear_cache()


class LanguageAwareModelWrapper:
    """
    Wrapper to make LanguageAwareModel compatible with trainers.
    """

    def __init__(
        self,
        base_model_wrapper: AbstractTrainableMTModel,
        language_embeddings_config: Optional[LanguageEmbeddingsConfig] = None,
    ):
        """
        Initialize wrapper.

        Args:
            base_model_wrapper: Base model wrapper
            language_embeddings_config: Language embeddings configuration
        """
        self.base_wrapper = base_model_wrapper
        self.language_embeddings_config = language_embeddings_config
        self.language_aware_model = None

    def enable_language_embeddings(self):
        """Enable language embeddings by wrapping the model."""
        if self.language_embeddings_config is None:
            logger.warning("No language embeddings config provided")
            return

        if not self.base_wrapper._loaded:
            raise RuntimeError("Base model not loaded. Call load() first.")

        logger.info("Enabling language embeddings...")

        # Create language-aware model
        self.language_aware_model = LanguageAwareModel(
            base_model=self.base_wrapper.model,
            language_embeddings_config=self.language_embeddings_config,
            source_lang=self.base_wrapper.config.source_lang,
            target_lang=self.base_wrapper.config.target_lang,
        )

        # Replace base model
        self.base_wrapper.model = self.language_aware_model

        logger.info("✓ Language embeddings enabled")

    def prepare_for_training(
        self,
        lora_config: LoRAConfig,
        training_config: TrainingConfig
    ):
        """Prepare model for training."""
        # Enable language embeddings before training
        if self.language_embeddings_config:
            self.enable_language_embeddings()

        # Prepare base model
        self.base_wrapper.prepare_for_training(lora_config, training_config)

    def __getattr__(self, name):
        """Delegate attribute access to base wrapper."""
        return getattr(self.base_wrapper, name)


def create_language_aware_model(
    base_model_class: type,
    model_config: ModelConfig,
    language_embeddings_config: Optional[LanguageEmbeddingsConfig] = None,
) -> AbstractTrainableMTModel:
    """
    Create model with optional language embeddings.

    Args:
        base_model_class: Base model class
        model_config: Model configuration
        language_embeddings_config: Language embeddings configuration

    Returns:
        Model wrapper (with or without language embeddings)
    """
    # Create base model
    base_model = base_model_class(model_config)

    # Wrap with language embeddings if config provided
    if language_embeddings_config and (
        language_embeddings_config.use_source_embedding or
        language_embeddings_config.use_target_embedding
    ):
        return LanguageAwareModelWrapper(base_model, language_embeddings_config)

    return base_model
