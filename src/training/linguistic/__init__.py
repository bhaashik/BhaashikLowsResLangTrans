"""
Linguistic features module for machine translation.

This module integrates linguistic knowledge into MT models:
- Dependency parsing (source/target/both sides)
- Feature extraction (deprel, POS, tree structure)
- Integration strategies (encoder augmentation, attention injection)

Usage:
    from src.training.linguistic import (
        StanzaParser,
        DependencyEncoder,
        LinguisticFeaturesConfig
    )

    # Parse texts
    parser = StanzaParser(language="hi")
    parses = parser.parse(["नमस्ते"])

    # Encode parse features
    encoder = DependencyEncoder(encoding_dim=128)
    features = encoder.encode_parse_tree(parses[0])

    # Use in training
    config = LinguisticFeaturesConfig(
        use_source_parse=True,
        parser="stanza",
        integration_method="encoder_augmentation"
    )
"""

from src.training.linguistic.parsers.base import (
    AbstractParser,
    ParseTree,
    ParserRegistry,
)

from src.training.linguistic.parsers.stanza_parser import StanzaParser

from src.training.linguistic.features.dependency_encoder import (
    DependencyEncoder,
    DependencyFeatures,
)

from src.training.linguistic.config import LinguisticFeaturesConfig

__all__ = [
    "AbstractParser",
    "ParseTree",
    "ParserRegistry",
    "StanzaParser",
    "DependencyEncoder",
    "DependencyFeatures",
    "LinguisticFeaturesConfig",
]
