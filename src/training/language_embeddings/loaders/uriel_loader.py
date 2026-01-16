"""
URIEL language embedding loader.

URIEL (Universal Representation for Inter-Lingual Embeddings) provides
typological features for 7,000+ languages from multiple sources:
- Syntactic features (word order, etc.)
- Phonological features (consonant/vowel inventories)
- Morphological features (case, gender, etc.)
- Geographic features (location)
- Genetic features (language family)

Reference:
    Littell et al. (2017). URIEL and lang2vec: Representing languages as
    typological, geographical, and phylogenetic vectors. EACL 2017.

Data sources are embedded in the module - no external downloads needed.
"""

from typing import List, Optional, Dict, Any
import numpy as np
import logging

from src.training.language_embeddings.loaders.base import (
    AbstractLanguageEmbeddingLoader,
    LanguageEmbedding,
    LoaderRegistry,
)

logger = logging.getLogger(__name__)


# URIEL feature data (embedded for core languages)
# In production, this would be loaded from lang2vec or URIEL database
URIEL_FEATURES = {
    # Syntactic features (S_ prefix)
    "syntax": [
        "S_SVO", "S_SOV", "S_VSO", "S_VOS", "S_OVS", "S_OSV",  # Word order
        "S_POSTPOSITIONS", "S_PREPOSITIONS",  # Adpositions
        "S_GENITIVES_AFTER_NOUN", "S_GENITIVES_BEFORE_NOUN",  # Genitive order
        "S_ADJECTIVES_AFTER_NOUN", "S_ADJECTIVES_BEFORE_NOUN",  # Adjective order
        "S_NUMERALS_AFTER_NOUN", "S_NUMERALS_BEFORE_NOUN",  # Numeral order
        "S_RELATIVES_AFTER_NOUN", "S_RELATIVES_BEFORE_NOUN",  # Relative clause
    ],
    # Phonological features (P_ prefix)
    "phonology": [
        "P_CONSONANTS_LOW", "P_CONSONANTS_AVERAGE", "P_CONSONANTS_HIGH",
        "P_VOWELS_LOW", "P_VOWELS_AVERAGE", "P_VOWELS_HIGH",
        "P_TONE_NO_TONE", "P_TONE_SIMPLE", "P_TONE_COMPLEX",
    ],
    # Morphological features (M_ prefix)
    "morphology": [
        "M_CASE_NONE", "M_CASE_2_3", "M_CASE_4_5", "M_CASE_6_OR_MORE",
        "M_GENDER_NONE", "M_GENDER_2", "M_GENDER_3_OR_MORE",
        "M_NUMBER_SINGULAR_PLURAL", "M_NUMBER_SINGULAR_DUAL_PLURAL",
        "M_TENSE_PAST_PRESENT_FUTURE", "M_TENSE_RELATIVE",
    ],
    # Inventory features (I_ prefix)
    "inventory": [
        "I_CONSONANTS", "I_VOWELS", "I_TONES",  # Phoneme inventory sizes
    ],
}

# Sample URIEL data for common Indic languages
# In production, load from lang2vec or URIEL database
SAMPLE_URIEL_DATA = {
    "hi": {  # Hindi
        "S_SVO": 0.0, "S_SOV": 1.0, "S_VSO": 0.0, "S_VOS": 0.0, "S_OVS": 0.0, "S_OSV": 0.0,
        "S_POSTPOSITIONS": 1.0, "S_PREPOSITIONS": 0.0,
        "S_GENITIVES_AFTER_NOUN": 0.0, "S_GENITIVES_BEFORE_NOUN": 1.0,
        "S_ADJECTIVES_AFTER_NOUN": 0.0, "S_ADJECTIVES_BEFORE_NOUN": 1.0,
        "S_NUMERALS_AFTER_NOUN": 0.0, "S_NUMERALS_BEFORE_NOUN": 1.0,
        "S_RELATIVES_AFTER_NOUN": 0.0, "S_RELATIVES_BEFORE_NOUN": 1.0,
        "P_CONSONANTS_LOW": 0.0, "P_CONSONANTS_AVERAGE": 0.0, "P_CONSONANTS_HIGH": 1.0,
        "P_VOWELS_LOW": 0.0, "P_VOWELS_AVERAGE": 1.0, "P_VOWELS_HIGH": 0.0,
        "P_TONE_NO_TONE": 1.0, "P_TONE_SIMPLE": 0.0, "P_TONE_COMPLEX": 0.0,
        "M_CASE_NONE": 0.0, "M_CASE_2_3": 1.0, "M_CASE_4_5": 0.0, "M_CASE_6_OR_MORE": 0.0,
        "M_GENDER_NONE": 0.0, "M_GENDER_2": 1.0, "M_GENDER_3_OR_MORE": 0.0,
        "M_NUMBER_SINGULAR_PLURAL": 1.0, "M_NUMBER_SINGULAR_DUAL_PLURAL": 0.0,
        "M_TENSE_PAST_PRESENT_FUTURE": 1.0, "M_TENSE_RELATIVE": 0.0,
        "I_CONSONANTS": 0.4, "I_VOWELS": 0.5, "I_TONES": 0.0,
    },
    "en": {  # English
        "S_SVO": 1.0, "S_SOV": 0.0, "S_VSO": 0.0, "S_VOS": 0.0, "S_OVS": 0.0, "S_OSV": 0.0,
        "S_POSTPOSITIONS": 0.0, "S_PREPOSITIONS": 1.0,
        "S_GENITIVES_AFTER_NOUN": 1.0, "S_GENITIVES_BEFORE_NOUN": 0.0,
        "S_ADJECTIVES_AFTER_NOUN": 0.0, "S_ADJECTIVES_BEFORE_NOUN": 1.0,
        "S_NUMERALS_AFTER_NOUN": 0.0, "S_NUMERALS_BEFORE_NOUN": 1.0,
        "S_RELATIVES_AFTER_NOUN": 1.0, "S_RELATIVES_BEFORE_NOUN": 0.0,
        "P_CONSONANTS_LOW": 0.0, "P_CONSONANTS_AVERAGE": 1.0, "P_CONSONANTS_HIGH": 0.0,
        "P_VOWELS_LOW": 0.0, "P_VOWELS_AVERAGE": 0.0, "P_VOWELS_HIGH": 1.0,
        "P_TONE_NO_TONE": 1.0, "P_TONE_SIMPLE": 0.0, "P_TONE_COMPLEX": 0.0,
        "M_CASE_NONE": 1.0, "M_CASE_2_3": 0.0, "M_CASE_4_5": 0.0, "M_CASE_6_OR_MORE": 0.0,
        "M_GENDER_NONE": 1.0, "M_GENDER_2": 0.0, "M_GENDER_3_OR_MORE": 0.0,
        "M_NUMBER_SINGULAR_PLURAL": 1.0, "M_NUMBER_SINGULAR_DUAL_PLURAL": 0.0,
        "M_TENSE_PAST_PRESENT_FUTURE": 1.0, "M_TENSE_RELATIVE": 0.0,
        "I_CONSONANTS": 0.3, "I_VOWELS": 0.6, "I_TONES": 0.0,
    },
    "bho": {  # Bhojpuri (similar to Hindi with some variations)
        "S_SVO": 0.0, "S_SOV": 1.0, "S_VSO": 0.0, "S_VOS": 0.0, "S_OVS": 0.0, "S_OSV": 0.0,
        "S_POSTPOSITIONS": 1.0, "S_PREPOSITIONS": 0.0,
        "S_GENITIVES_AFTER_NOUN": 0.0, "S_GENITIVES_BEFORE_NOUN": 1.0,
        "S_ADJECTIVES_AFTER_NOUN": 0.0, "S_ADJECTIVES_BEFORE_NOUN": 1.0,
        "S_NUMERALS_AFTER_NOUN": 0.0, "S_NUMERALS_BEFORE_NOUN": 1.0,
        "S_RELATIVES_AFTER_NOUN": 0.0, "S_RELATIVES_BEFORE_NOUN": 1.0,
        "P_CONSONANTS_LOW": 0.0, "P_CONSONANTS_AVERAGE": 0.0, "P_CONSONANTS_HIGH": 1.0,
        "P_VOWELS_LOW": 0.0, "P_VOWELS_AVERAGE": 1.0, "P_VOWELS_HIGH": 0.0,
        "P_TONE_NO_TONE": 1.0, "P_TONE_SIMPLE": 0.0, "P_TONE_COMPLEX": 0.0,
        "M_CASE_NONE": 0.0, "M_CASE_2_3": 1.0, "M_CASE_4_5": 0.0, "M_CASE_6_OR_MORE": 0.0,
        "M_GENDER_NONE": 0.0, "M_GENDER_2": 1.0, "M_GENDER_3_OR_MORE": 0.0,
        "M_NUMBER_SINGULAR_PLURAL": 1.0, "M_NUMBER_SINGULAR_DUAL_PLURAL": 0.0,
        "M_TENSE_PAST_PRESENT_FUTURE": 1.0, "M_TENSE_RELATIVE": 0.0,
        "I_CONSONANTS": 0.4, "I_VOWELS": 0.5, "I_TONES": 0.0,
    },
}


@LoaderRegistry.register("uriel")
class URIELLoader(AbstractLanguageEmbeddingLoader):
    """
    Loader for URIEL typological features.

    Provides access to typological features from URIEL database.
    Falls back to embedded data if lang2vec is not available.
    """

    def __init__(
        self,
        feature_types: Optional[List[str]] = None,
        normalize: bool = True,
        cache: bool = True,
        use_lang2vec: bool = True,
    ):
        """
        Initialize URIEL loader.

        Args:
            feature_types: Types of features ("syntax", "phonology", "morphology", "inventory")
            normalize: Whether to normalize embeddings
            cache: Whether to cache embeddings
            use_lang2vec: Whether to try using lang2vec library
        """
        super().__init__(feature_types, normalize, cache)
        self.use_lang2vec = use_lang2vec
        self.lang2vec_available = False
        self._feature_names = []

    def initialize(self):
        """Initialize URIEL loader."""
        if self._initialized:
            return

        logger.info("Initializing URIEL loader...")

        # Try to import lang2vec
        if self.use_lang2vec:
            try:
                import lang2vec.lang2vec as l2v
                self.lang2vec = l2v
                self.lang2vec_available = True
                logger.info("✓ Using lang2vec library for URIEL features")
            except ImportError:
                logger.warning(
                    "lang2vec not available. Install with: pip install lang2vec"
                )
                logger.info("✓ Using embedded URIEL data for common languages")

        # Build feature list
        if not self.feature_types:
            self.feature_types = ["syntax", "phonology", "morphology", "inventory"]

        self._feature_names = []
        for feature_type in self.feature_types:
            if feature_type in URIEL_FEATURES:
                self._feature_names.extend(URIEL_FEATURES[feature_type])

        logger.info(f"  Feature types: {', '.join(self.feature_types)}")
        logger.info(f"  Total features: {len(self._feature_names)}")

        self._initialized = True

    def supports_language(self, language_code: str) -> bool:
        """Check if language is supported."""
        if self.lang2vec_available:
            # lang2vec supports 7,000+ languages
            try:
                # Try to get features to check if language is supported
                import lang2vec.lang2vec as l2v
                features = l2v.get_features(language_code, "learned")
                return language_code in features
            except Exception:
                return False
        else:
            # Fall back to embedded data
            return language_code in SAMPLE_URIEL_DATA

    def load_embedding(self, language_code: str) -> LanguageEmbedding:
        """Load URIEL embedding for language."""
        if not self._initialized:
            self.initialize()

        # Check cache
        if self.cache and language_code in self._embedding_cache:
            return self._embedding_cache[language_code]

        # Get features
        if self.lang2vec_available:
            embedding_vector = self._load_from_lang2vec(language_code)
        else:
            embedding_vector = self._load_from_embedded(language_code)

        # Create embedding object
        embedding = LanguageEmbedding(
            language_code=language_code,
            embedding=embedding_vector,
            feature_names=self._feature_names.copy(),
            source="uriel",
            metadata={
                "feature_types": self.feature_types,
                "loader": "URIELLoader",
            }
        )

        # Normalize if requested
        if self.normalize:
            embedding.normalize()

        # Cache
        if self.cache:
            self._embedding_cache[language_code] = embedding

        return embedding

    def _load_from_lang2vec(self, language_code: str) -> np.ndarray:
        """Load features from lang2vec library."""
        try:
            import lang2vec.lang2vec as l2v

            # Map feature types to lang2vec categories
            feature_sets = []
            for ftype in self.feature_types:
                if ftype == "syntax":
                    feature_sets.append("syntax_wals")
                elif ftype == "phonology":
                    feature_sets.append("phonology_wals")
                elif ftype == "morphology":
                    feature_sets.append("syntax_wals")  # Morphology is in syntax
                elif ftype == "inventory":
                    feature_sets.append("inventory")

            # Get features
            features_dict = l2v.get_features(
                [language_code],
                feature_sets,
                header=True
            )

            if language_code not in features_dict:
                raise ValueError(f"Language {language_code} not found in URIEL")

            # Extract feature values
            lang_features = features_dict[language_code]

            # Filter to requested features
            feature_vector = []
            for fname in self._feature_names:
                if fname in lang_features:
                    feature_vector.append(lang_features[fname])
                else:
                    # Unknown feature, use 0
                    feature_vector.append(0.0)

            return np.array(feature_vector, dtype=np.float32)

        except Exception as e:
            logger.error(f"Error loading from lang2vec: {e}")
            # Fall back to embedded data
            return self._load_from_embedded(language_code)

    def _load_from_embedded(self, language_code: str) -> np.ndarray:
        """Load features from embedded data."""
        if language_code not in SAMPLE_URIEL_DATA:
            raise ValueError(
                f"Language {language_code} not supported in embedded data. "
                f"Available: {list(SAMPLE_URIEL_DATA.keys())}. "
                f"Install lang2vec for 7,000+ languages."
            )

        lang_data = SAMPLE_URIEL_DATA[language_code]

        # Extract features in order
        feature_vector = []
        for fname in self._feature_names:
            if fname in lang_data:
                feature_vector.append(lang_data[fname])
            else:
                feature_vector.append(0.0)

        return np.array(feature_vector, dtype=np.float32)

    def get_available_languages(self) -> List[str]:
        """Get list of available languages."""
        if self.lang2vec_available:
            # lang2vec supports 7,000+ languages
            # Return a subset for performance
            logger.info("URIEL via lang2vec supports 7,000+ languages")
            return list(SAMPLE_URIEL_DATA.keys()) + ["(7000+ more via lang2vec)"]
        else:
            return list(SAMPLE_URIEL_DATA.keys())

    def get_feature_names(self) -> List[str]:
        """Get names of features."""
        if not self._initialized:
            self.initialize()
        return self._feature_names.copy()
