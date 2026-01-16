"""
Encoder augmentation: Concatenate parse features with word embeddings.

This is the simplest integration strategy. Parse features are extracted,
encoded, and concatenated with word embeddings before passing to the encoder.

Architecture:
    [word_embedding; parse_features] → Encoder → Decoder
"""

from typing import List, Optional, Dict, Any
import logging

from src.training.linguistic.parsers.base import ParseTree, create_parser
from src.training.linguistic.features.dependency_encoder import DependencyEncoder
from src.training.linguistic.config import LinguisticFeaturesConfig

logger = logging.getLogger(__name__)


class EncoderAugmentation:
    """
    Encoder augmentation with linguistic features.

    Augments encoder inputs by concatenating parse features
    with word embeddings.
    """

    def __init__(
        self,
        config: LinguisticFeaturesConfig,
        source_lang: str,
        target_lang: Optional[str] = None,
    ):
        """
        Initialize encoder augmentation.

        Args:
            config: Linguistic features configuration
            source_lang: Source language code
            target_lang: Target language code (if using target-side features)
        """
        self.config = config
        self.source_lang = source_lang
        self.target_lang = target_lang

        # Initialize parsers
        self.source_parser = None
        self.target_parser = None

        if config.use_source_parse:
            self.source_parser = create_parser(
                config.parser,
                source_lang,
                use_gpu=config.parser_use_gpu,
                batch_size=config.parser_batch_size
            )
            logger.info(f"Initialized source parser for {source_lang}")

        if config.use_target_parse and target_lang:
            self.target_parser = create_parser(
                config.parser,
                target_lang,
                use_gpu=config.parser_use_gpu,
                batch_size=config.parser_batch_size
            )
            logger.info(f"Initialized target parser for {target_lang}")

        # Initialize feature encoder
        self.feature_encoder = DependencyEncoder(
            encoding_dim=config.encoding_dim,
            use_graph_encoder=config.use_graph_encoder,
            feature_types=config.features,
        )

        # Parse cache
        self.parse_cache: Dict[str, ParseTree] = {}

    def parse_texts(
        self,
        texts: List[str],
        language: str,
        use_cache: bool = True
    ) -> List[ParseTree]:
        """
        Parse texts.

        Args:
            texts: Texts to parse
            language: Language code
            use_cache: Whether to use cache

        Returns:
            List of ParseTree objects
        """
        # Select parser
        if language == self.source_lang:
            parser = self.source_parser
        elif language == self.target_lang:
            parser = self.target_parser
        else:
            raise ValueError(f"No parser for language: {language}")

        if parser is None:
            raise ValueError(f"Parser not initialized for {language}")

        # Check cache
        if use_cache and self.config.cache_parses:
            uncached_texts = []
            uncached_indices = []

            for i, text in enumerate(texts):
                cache_key = f"{language}:{text}"
                if cache_key not in self.parse_cache:
                    uncached_texts.append(text)
                    uncached_indices.append(i)

            # Parse uncached texts
            if uncached_texts:
                logger.debug(f"Parsing {len(uncached_texts)} uncached texts")
                uncached_parses = parser.parse(uncached_texts)

                # Cache results
                for text, parse in zip(uncached_texts, uncached_parses):
                    cache_key = f"{language}:{text}"
                    self.parse_cache[cache_key] = parse

            # Retrieve all from cache
            parses = []
            for text in texts:
                cache_key = f"{language}:{text}"
                parses.append(self.parse_cache[cache_key])

            return parses

        # Parse without cache
        return parser.parse(texts)

    def encode_features(
        self,
        parses: List[ParseTree],
        return_tensor: bool = True
    ):
        """
        Encode parse features.

        Args:
            parses: List of parse trees
            return_tensor: Whether to return torch.Tensor

        Returns:
            Encoded features (tensor or numpy array)
        """
        return self.feature_encoder.encode_batch(parses, return_tensor=return_tensor)

    def augment_source(
        self,
        source_texts: List[str],
        source_embeddings,  # torch.Tensor [batch, seq_len, emb_dim]
        use_cache: bool = True
    ):
        """
        Augment source embeddings with parse features.

        Args:
            source_texts: Source texts
            source_embeddings: Source word embeddings [batch, seq_len, emb_dim]
            use_cache: Whether to use parse cache

        Returns:
            Augmented embeddings [batch, seq_len, emb_dim + encoding_dim]
        """
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch required for encoder augmentation")

        # Parse source texts
        source_parses = self.parse_texts(
            source_texts,
            self.source_lang,
            use_cache=use_cache
        )

        # Encode features
        source_features = self.encode_features(source_parses, return_tensor=True)

        # Ensure same sequence length (pad/truncate if needed)
        batch_size, seq_len, emb_dim = source_embeddings.shape
        feature_seq_len = source_features.shape[1]

        if feature_seq_len < seq_len:
            # Pad features
            padding = torch.zeros(
                batch_size,
                seq_len - feature_seq_len,
                source_features.shape[2],
                device=source_features.device
            )
            source_features = torch.cat([source_features, padding], dim=1)
        elif feature_seq_len > seq_len:
            # Truncate features
            source_features = source_features[:, :seq_len, :]

        # Concatenate embeddings and features
        augmented = torch.cat([source_embeddings, source_features], dim=2)

        return augmented

    def cleanup(self):
        """Cleanup parsers and cache."""
        if self.source_parser:
            self.source_parser.cleanup()
        if self.target_parser:
            self.target_parser.cleanup()

        self.parse_cache.clear()


def augment_encoder_inputs(
    source_texts: List[str],
    source_embeddings,
    source_lang: str,
    config: LinguisticFeaturesConfig,
    augmentation: Optional[EncoderAugmentation] = None
):
    """
    Convenience function to augment encoder inputs.

    Args:
        source_texts: Source texts
        source_embeddings: Source embeddings tensor
        source_lang: Source language code
        config: Linguistic features config
        augmentation: Pre-initialized EncoderAugmentation (creates new if None)

    Returns:
        Augmented embeddings
    """
    if augmentation is None:
        augmentation = EncoderAugmentation(config, source_lang)

    return augmentation.augment_source(source_texts, source_embeddings)


# Example usage
def example_usage():
    """Example of using encoder augmentation."""
    import torch

    # Configuration
    config = LinguisticFeaturesConfig(
        use_source_parse=True,
        parser="stanza",
        features=["dependency_labels", "pos_tags"],
        integration_method="encoder_augmentation",
        encoding_dim=128,
    )

    # Create augmentation
    augmentation = EncoderAugmentation(config, source_lang="hi")

    # Source texts and embeddings
    source_texts = ["नमस्ते", "आप कैसे हैं?"]
    source_embeddings = torch.randn(2, 10, 512)  # [batch=2, seq=10, dim=512]

    # Augment
    augmented = augmentation.augment_source(source_texts, source_embeddings)

    print(f"Original shape: {source_embeddings.shape}")
    print(f"Augmented shape: {augmented.shape}")
    # Output: Augmented shape: torch.Size([2, 10, 640])  # 512 + 128
