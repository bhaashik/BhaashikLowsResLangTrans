"""
Stanza parser implementation.

Stanza is a Python NLP library that provides state-of-the-art neural models
for Universal Dependencies parsing in 60+ languages.

Reference: https://stanfordnlp.github.io/stanza/
"""

from typing import List, Optional
import logging

from src.training.linguistic.parsers.base import (
    AbstractParser,
    ParseTree,
    ParserRegistry,
)

logger = logging.getLogger(__name__)


@ParserRegistry.register("stanza")
class StanzaParser(AbstractParser):
    """
    Stanza-based dependency parser.

    Supports 60+ languages with pre-trained UD models.
    """

    # Language code mapping (ISO 639 → Stanza)
    LANGUAGE_MAP = {
        # Indic languages
        "hi": "hi",  # Hindi
        "bn": "bn",  # Bengali
        "te": "te",  # Telugu
        "ta": "ta",  # Tamil
        "mr": "mr",  # Marathi
        "gu": "gu",  # Gujarati
        "ur": "ur",  # Urdu
        "ml": "ml",  # Malayalam
        "kn": "kn",  # Kannada
        "or": "or",  # Oriya
        "pa": "pa",  # Punjabi
        "sa": "sa",  # Sanskrit
        # Other languages
        "en": "en",  # English
        "zh": "zh-hans",  # Chinese (Simplified)
        "es": "es",  # Spanish
        "fr": "fr",  # French
        "de": "de",  # German
        "ru": "ru",  # Russian
        "ar": "ar",  # Arabic
        "ja": "ja",  # Japanese
        "ko": "ko",  # Korean
    }

    def __init__(
        self,
        language: str,
        use_gpu: bool = True,
        batch_size: int = 32,
        **kwargs
    ):
        """
        Initialize Stanza parser.

        Args:
            language: ISO 639 language code
            use_gpu: Whether to use GPU
            batch_size: Batch size for parsing
            **kwargs: Additional Stanza parameters
        """
        super().__init__(language, **kwargs)
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.pipeline = None

    def initialize(self):
        """Initialize Stanza pipeline."""
        if self._initialized:
            return

        try:
            import stanza

            # Map language code
            stanza_lang = self._get_stanza_language(self.language)

            logger.info(f"Initializing Stanza parser for {stanza_lang}")

            # Download model if needed
            try:
                stanza.download(stanza_lang, verbose=False)
            except Exception as e:
                logger.warning(f"Failed to download Stanza model: {e}")

            # Initialize pipeline
            self.pipeline = stanza.Pipeline(
                stanza_lang,
                processors='tokenize,pos,lemma,depparse',
                tokenize_pretokenized=False,  # We pass raw text
                use_gpu=self.use_gpu,
                verbose=False,
                **self.kwargs
            )

            self._initialized = True
            logger.info(f"Stanza parser initialized for {stanza_lang}")

        except ImportError:
            raise ImportError(
                "Stanza not installed. Install with: pip install stanza"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Stanza parser: {e}")

    def parse(
        self,
        texts: List[str],
        language: Optional[str] = None
    ) -> List[ParseTree]:
        """
        Parse texts using Stanza.

        Args:
            texts: Texts to parse
            language: Language override (not used, parser is language-specific)

        Returns:
            List of ParseTree objects
        """
        if not self._initialized:
            self.initialize()

        if not texts:
            return []

        logger.debug(f"Parsing {len(texts)} texts with Stanza")

        parse_trees = []

        # Process in batches for efficiency
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            for text in batch:
                try:
                    # Parse with Stanza
                    doc = self.pipeline(text)

                    # Extract parse trees for each sentence
                    for sentence in doc.sentences:
                        parse_tree = self._convert_sentence(sentence)
                        parse_trees.append(parse_tree)

                except Exception as e:
                    logger.error(f"Failed to parse text: {text[:50]}... Error: {e}")
                    # Return empty parse tree as fallback
                    words = text.split()
                    parse_trees.append(ParseTree(
                        words=words,
                        heads=[0] * len(words),
                        deprels=["dep"] * len(words),
                        pos_tags=["X"] * len(words),
                        lemmas=words,
                        metadata={'error': str(e)}
                    ))

        return parse_trees

    def _convert_sentence(self, sentence) -> ParseTree:
        """
        Convert Stanza sentence to ParseTree.

        Args:
            sentence: Stanza sentence object

        Returns:
            ParseTree object
        """
        words = []
        heads = []
        deprels = []
        pos_tags = []
        lemmas = []
        features = []

        for word in sentence.words:
            words.append(word.text)
            heads.append(word.head)  # Already 0-indexed for root, 1-indexed for others
            deprels.append(word.deprel if word.deprel else "dep")
            pos_tags.append(word.upos if word.upos else "X")
            lemmas.append(word.lemma if word.lemma else word.text)

            # Parse morphological features
            if word.feats:
                feat_dict = {}
                for feat in word.feats.split("|"):
                    if "=" in feat:
                        key, value = feat.split("=", 1)
                        feat_dict[key] = value
                features.append(feat_dict)
            else:
                features.append({})

        return ParseTree(
            words=words,
            heads=heads,
            deprels=deprels,
            pos_tags=pos_tags,
            lemmas=lemmas,
            features=features,
            metadata={
                'language': self.language,
                'parser': 'stanza'
            }
        )

    def supports_language(self, language: str) -> bool:
        """Check if Stanza supports language."""
        stanza_lang = self._get_stanza_language(language)
        return stanza_lang is not None

    def _get_stanza_language(self, language: str) -> Optional[str]:
        """
        Map ISO 639 code to Stanza language code.

        Args:
            language: ISO 639 code

        Returns:
            Stanza language code or None if not supported
        """
        # Direct mapping
        if language in self.LANGUAGE_MAP:
            return self.LANGUAGE_MAP[language]

        # Try language as-is (Stanza might support it)
        return language

    def cleanup(self):
        """Cleanup Stanza resources."""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None

        # Force garbage collection
        import gc
        gc.collect()

        if self.use_gpu:
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass

        super().cleanup()


def parse_with_stanza(
    texts: List[str],
    language: str,
    use_gpu: bool = True,
    batch_size: int = 32
) -> List[ParseTree]:
    """
    Convenience function to parse texts with Stanza.

    Args:
        texts: Texts to parse
        language: Language code
        use_gpu: Whether to use GPU
        batch_size: Batch size

    Returns:
        List of ParseTree objects
    """
    with StanzaParser(language, use_gpu=use_gpu, batch_size=batch_size) as parser:
        return parser.parse(texts)
