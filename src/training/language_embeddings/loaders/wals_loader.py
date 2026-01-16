"""
WALS (World Atlas of Language Structures) loader.

WALS provides typological features for 2,000+ languages based on linguistic
research. Features include word order, morphology, phonology, and more.

Reference:
    Dryer, Matthew S. & Haspelmath, Martin (eds.) 2013.
    The World Atlas of Language Structures Online.
    Leipzig: Max Planck Institute for Evolutionary Anthropology.
    https://wals.info/

Note: This is a simplified implementation. For production use, integrate with
the full WALS database or use the pywals library.
"""

from typing import List, Optional
import numpy as np
import logging

from src.training.language_embeddings.loaders.base import (
    AbstractLanguageEmbeddingLoader,
    LanguageEmbedding,
    LoaderRegistry,
)

logger = logging.getLogger(__name__)


# Sample WALS features (subset)
WALS_FEATURES = {
    "syntax": [
        "81A",  # Order of Subject, Object and Verb
        "82A",  # Order of Subject and Verb
        "83A",  # Order of Object and Verb
        "85A",  # Order of Adposition and Noun Phrase
        "86A",  # Order of Genitive and Noun
        "87A",  # Order of Adjective and Noun
        "88A",  # Order of Demonstrative and Noun
        "89A",  # Order of Numeral and Noun
        "90A",  # Order of Relative Clause and Noun
    ],
    "phonology": [
        "1A",   # Consonant Inventories
        "2A",   # Vowel Quality Inventories
        "3A",   # Consonant-Vowel Ratio
        "13A",  # Tone
    ],
    "morphology": [
        "49A",  # Number of Cases
        "30A",  # Number of Genders
        "33A",  # Coding of Nominal Plurality
        "21A",  # Exponence of Selected Inflectional Formatives
    ],
}

# Sample WALS data for common languages
# Values are categorical (encoded as integers)
SAMPLE_WALS_DATA = {
    "hi": {  # Hindi
        "81A": 6,   # SOV
        "82A": 1,   # SV
        "83A": 2,   # OV
        "85A": 2,   # Postpositions
        "86A": 2,   # Genitive-Noun
        "87A": 2,   # Adjective-Noun
        "88A": 1,   # Demonstrative-Noun
        "89A": 2,   # Numeral-Noun
        "90A": 2,   # Relative-Noun
        "1A": 3,    # Average consonants (22)
        "2A": 3,    # Average vowels (7-14)
        "3A": 2,    # Moderately low
        "13A": 1,   # No tone
        "49A": 2,   # 2-5 cases
        "30A": 2,   # 2 genders
        "33A": 2,   # Plural suffix
        "21A": 2,   # Synthetic
    },
    "en": {  # English
        "81A": 1,   # SVO
        "82A": 1,   # SV
        "83A": 1,   # VO
        "85A": 1,   # Prepositions
        "86A": 1,   # Noun-Genitive
        "87A": 2,   # Adjective-Noun
        "88A": 1,   # Demonstrative-Noun
        "89A": 2,   # Numeral-Noun
        "90A": 1,   # Noun-Relative
        "1A": 3,    # Average consonants (24)
        "2A": 3,    # Average vowels (7-14)
        "3A": 2,    # Moderately low
        "13A": 1,   # No tone
        "49A": 1,   # 0-1 cases
        "30A": 1,   # No gender
        "33A": 2,   # Plural suffix
        "21A": 2,   # Synthetic
    },
    "bho": {  # Bhojpuri (similar to Hindi)
        "81A": 6,   # SOV
        "82A": 1,   # SV
        "83A": 2,   # OV
        "85A": 2,   # Postpositions
        "86A": 2,   # Genitive-Noun
        "87A": 2,   # Adjective-Noun
        "88A": 1,   # Demonstrative-Noun
        "89A": 2,   # Numeral-Noun
        "90A": 2,   # Relative-Noun
        "1A": 3,    # Average consonants
        "2A": 3,    # Average vowels
        "3A": 2,    # Moderately low
        "13A": 1,   # No tone
        "49A": 2,   # 2-5 cases
        "30A": 2,   # 2 genders
        "33A": 2,   # Plural suffix
        "21A": 2,   # Synthetic
    },
}


@LoaderRegistry.register("wals")
class WALSLoader(AbstractLanguageEmbeddingLoader):
    """
    Loader for WALS typological features.

    Provides access to typological features from World Atlas of Language Structures.
    Falls back to embedded data if pywals is not available.
    """

    def __init__(
        self,
        feature_types: Optional[List[str]] = None,
        normalize: bool = True,
        cache: bool = True,
        use_pywals: bool = True,
    ):
        """
        Initialize WALS loader.

        Args:
            feature_types: Types of features ("syntax", "phonology", "morphology")
            normalize: Whether to normalize embeddings
            cache: Whether to cache embeddings
            use_pywals: Whether to try using pywals library
        """
        super().__init__(feature_types, normalize, cache)
        self.use_pywals = use_pywals
        self.pywals_available = False
        self._feature_names = []
        self._max_values = {}  # For normalization

    def initialize(self):
        """Initialize WALS loader."""
        if self._initialized:
            return

        logger.info("Initializing WALS loader...")

        # Try to import pywals
        if self.use_pywals:
            try:
                import pywals
                self.pywals = pywals
                self.pywals_available = True
                logger.info("✓ Using pywals library for WALS features")
            except ImportError:
                logger.warning(
                    "pywals not available. Install with: pip install pywals"
                )
                logger.info("✓ Using embedded WALS data for common languages")

        # Build feature list
        if not self.feature_types:
            self.feature_types = ["syntax", "phonology", "morphology"]

        self._feature_names = []
        for feature_type in self.feature_types:
            if feature_type in WALS_FEATURES:
                self._feature_names.extend(WALS_FEATURES[feature_type])

        # Compute max values for normalization
        for fname in self._feature_names:
            max_val = 0
            for lang_data in SAMPLE_WALS_DATA.values():
                if fname in lang_data:
                    max_val = max(max_val, lang_data[fname])
            self._max_values[fname] = max_val if max_val > 0 else 1

        logger.info(f"  Feature types: {', '.join(self.feature_types)}")
        logger.info(f"  Total features: {len(self._feature_names)}")

        self._initialized = True

    def supports_language(self, language_code: str) -> bool:
        """Check if language is supported."""
        if self.pywals_available:
            # Check if language exists in WALS
            try:
                # pywals API check would go here
                return True
            except Exception:
                return False
        else:
            # Fall back to embedded data
            return language_code in SAMPLE_WALS_DATA

    def load_embedding(self, language_code: str) -> LanguageEmbedding:
        """Load WALS embedding for language."""
        if not self._initialized:
            self.initialize()

        # Check cache
        if self.cache and language_code in self._embedding_cache:
            return self._embedding_cache[language_code]

        # Get features
        if self.pywals_available:
            embedding_vector = self._load_from_pywals(language_code)
        else:
            embedding_vector = self._load_from_embedded(language_code)

        # Create embedding object
        embedding = LanguageEmbedding(
            language_code=language_code,
            embedding=embedding_vector,
            feature_names=self._feature_names.copy(),
            source="wals",
            metadata={
                "feature_types": self.feature_types,
                "loader": "WALSLoader",
            }
        )

        # Normalize if requested
        if self.normalize:
            embedding.normalize()

        # Cache
        if self.cache:
            self._embedding_cache[language_code] = embedding

        return embedding

    def _load_from_pywals(self, language_code: str) -> np.ndarray:
        """Load features from pywals library."""
        # Placeholder for pywals integration
        # In production, use pywals API to fetch features
        logger.warning("pywals integration not yet implemented, using embedded data")
        return self._load_from_embedded(language_code)

    def _load_from_embedded(self, language_code: str) -> np.ndarray:
        """Load features from embedded data."""
        if language_code not in SAMPLE_WALS_DATA:
            raise ValueError(
                f"Language {language_code} not supported in embedded WALS data. "
                f"Available: {list(SAMPLE_WALS_DATA.keys())}. "
                f"Install pywals for 2,000+ languages."
            )

        lang_data = SAMPLE_WALS_DATA[language_code]

        # Extract and normalize features
        feature_vector = []
        for fname in self._feature_names:
            if fname in lang_data:
                # Normalize categorical values to [0, 1]
                raw_value = lang_data[fname]
                normalized_value = raw_value / self._max_values.get(fname, 1)
                feature_vector.append(normalized_value)
            else:
                # Missing feature
                feature_vector.append(0.0)

        return np.array(feature_vector, dtype=np.float32)

    def get_available_languages(self) -> List[str]:
        """Get list of available languages."""
        if self.pywals_available:
            logger.info("WALS via pywals supports 2,000+ languages")
            return list(SAMPLE_WALS_DATA.keys()) + ["(2000+ more via pywals)"]
        else:
            return list(SAMPLE_WALS_DATA.keys())

    def get_feature_names(self) -> List[str]:
        """Get names of features."""
        if not self._initialized:
            self.initialize()
        return self._feature_names.copy()
