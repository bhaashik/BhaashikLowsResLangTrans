"""Dependency parsers for linguistic feature extraction."""

from src.training.linguistic.parsers.base import (
    AbstractParser,
    ParseTree,
    ParserRegistry,
)

from src.training.linguistic.parsers.stanza_parser import StanzaParser

__all__ = [
    "AbstractParser",
    "ParseTree",
    "ParserRegistry",
    "StanzaParser",
]
