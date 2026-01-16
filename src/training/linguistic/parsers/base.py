"""
Base parser interface for dependency parsing.

Provides abstract interface that all parsers must implement,
plus common ParseTree representation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Type
import logging

logger = logging.getLogger(__name__)


@dataclass
class ParseTree:
    """
    Dependency parse tree representation.

    Follows Universal Dependencies (UD) conventions where applicable.

    Attributes:
        words: List of word forms
        heads: Head indices (0 = root, 1-indexed for words)
        deprels: Dependency relation labels (e.g., "nsubj", "obj")
        pos_tags: Universal POS tags (e.g., "NOUN", "VERB")
        lemmas: Lemmatized word forms
        features: Morphological features (UD format)
        metadata: Additional metadata (language, sentence_id, etc.)
    """
    words: List[str]
    heads: List[int]
    deprels: List[str]
    pos_tags: List[str]
    lemmas: Optional[List[str]] = None
    features: Optional[List[Dict[str, str]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate parse tree structure."""
        n = len(self.words)
        if len(self.heads) != n:
            raise ValueError(f"heads length ({len(self.heads)}) != words length ({n})")
        if len(self.deprels) != n:
            raise ValueError(f"deprels length ({len(self.deprels)}) != words length ({n})")
        if len(self.pos_tags) != n:
            raise ValueError(f"pos_tags length ({len(self.pos_tags)}) != words length ({n})")

    def get_tree_depth(self, word_idx: int) -> int:
        """
        Get depth of word in tree (distance to root).

        Args:
            word_idx: Word index (0-indexed)

        Returns:
            Depth (0 = root, 1 = child of root, etc.)
        """
        depth = 0
        current = word_idx
        visited = set()

        while self.heads[current] != 0:
            if current in visited:
                # Cycle detected (malformed tree)
                logger.warning(f"Cycle detected in parse tree at word {current}")
                return -1
            visited.add(current)
            current = self.heads[current] - 1  # Convert to 0-indexed
            depth += 1

        return depth

    def get_children(self, word_idx: int) -> List[int]:
        """
        Get children of word in tree.

        Args:
            word_idx: Word index (0-indexed)

        Returns:
            List of child indices (0-indexed)
        """
        return [
            i for i, head in enumerate(self.heads)
            if head == word_idx + 1  # heads are 1-indexed
        ]

    def get_subtree(self, word_idx: int) -> List[int]:
        """
        Get all descendants of word (subtree).

        Args:
            word_idx: Word index (0-indexed)

        Returns:
            List of descendant indices (0-indexed), including word_idx
        """
        subtree = [word_idx]
        children = self.get_children(word_idx)

        for child in children:
            subtree.extend(self.get_subtree(child))

        return subtree

    def to_conllu(self) -> str:
        """
        Convert parse tree to CoNLL-U format.

        Returns:
            CoNLL-U formatted string
        """
        lines = []
        for i in range(len(self.words)):
            fields = [
                str(i + 1),  # ID
                self.words[i],  # FORM
                self.lemmas[i] if self.lemmas else "_",  # LEMMA
                self.pos_tags[i],  # UPOS
                "_",  # XPOS
                "_",  # FEATS (could add self.features[i])
                str(self.heads[i]),  # HEAD
                self.deprels[i],  # DEPREL
                "_",  # DEPS
                "_",  # MISC
            ]
            lines.append("\t".join(fields))

        return "\n".join(lines)


class AbstractParser(ABC):
    """
    Abstract interface for dependency parsers.

    All parser implementations must inherit from this class.
    """

    def __init__(self, language: str, **kwargs):
        """
        Initialize parser.

        Args:
            language: ISO 639 language code (e.g., "hi", "en", "bho")
            **kwargs: Parser-specific parameters
        """
        self.language = language
        self.kwargs = kwargs
        self._initialized = False

    @abstractmethod
    def parse(self, texts: List[str], language: Optional[str] = None) -> List[ParseTree]:
        """
        Parse texts and return dependency trees.

        Args:
            texts: Texts to parse
            language: Language override (uses self.language if None)

        Returns:
            List of ParseTree objects
        """
        pass

    @abstractmethod
    def supports_language(self, language: str) -> bool:
        """
        Check if parser supports language.

        Args:
            language: ISO 639 language code

        Returns:
            True if language is supported
        """
        pass

    @abstractmethod
    def initialize(self):
        """
        Initialize parser (load models, etc.).

        This is called once before first use.
        """
        pass

    def cleanup(self):
        """
        Cleanup parser resources.

        Override if needed (e.g., to unload models).
        """
        self._initialized = False

    def __enter__(self):
        """Context manager entry."""
        if not self._initialized:
            self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


class ParserRegistry:
    """
    Registry for dependency parsers.

    Allows dynamic registration and retrieval of parsers.
    """

    _parsers: Dict[str, Type[AbstractParser]] = {}

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a parser.

        Usage:
            @ParserRegistry.register("stanza")
            class StanzaParser(AbstractParser):
                ...
        """
        def decorator(parser_class: Type[AbstractParser]):
            cls._parsers[name] = parser_class
            logger.info(f"Registered parser: {name}")
            return parser_class
        return decorator

    @classmethod
    def get_parser(
        cls,
        name: str,
        language: str,
        **kwargs
    ) -> AbstractParser:
        """
        Get parser instance by name.

        Args:
            name: Parser name (e.g., "stanza", "spacy")
            language: Language code
            **kwargs: Parser-specific parameters

        Returns:
            Parser instance
        """
        if name not in cls._parsers:
            raise ValueError(
                f"Unknown parser: {name}. "
                f"Available: {list(cls._parsers.keys())}"
            )

        parser_class = cls._parsers[name]
        return parser_class(language=language, **kwargs)

    @classmethod
    def list_parsers(cls) -> List[str]:
        """List available parsers."""
        return list(cls._parsers.keys())


def create_parser(
    parser_type: str,
    language: str,
    **kwargs
) -> AbstractParser:
    """
    Convenience function to create parser.

    Args:
        parser_type: Parser name
        language: Language code
        **kwargs: Parser parameters

    Returns:
        Initialized parser instance
    """
    parser = ParserRegistry.get_parser(parser_type, language, **kwargs)
    parser.initialize()
    return parser
