"""
Ensemble methods for combining translations from multiple models.

This module provides various strategies for combining outputs from multiple
translation models to improve translation quality:

- COMET-based selection: Use COMET metric to select best translations
- MBR (Minimum Bayes Risk): Select translations that minimize expected loss
- Voting methods: Majority voting, weighted voting
- Reranking: Use quality estimation models to rerank candidates
- Bilingual scoring: Use bilingual embeddings to score translations
- Back-translation: Validate translations via round-trip translation

Usage:
    from src.training.ensemble import COMETEnsemble, MBREnsemble

    # COMET-based selection
    ensemble = COMETEnsemble(models=[model1, model2, model3])
    result = ensemble.translate(["Hello"], source_lang="en", target_lang="hi")

    # MBR decoding
    mbr = MBREnsemble(models=[model1, model2, model3], utility_metric="bleu")
    result = mbr.translate(["Hello"], source_lang="en", target_lang="hi")
"""

from src.training.ensemble.base import (
    AbstractEnsemble,
    EnsembleResult,
    EnsembleConfig,
)

from src.training.ensemble.comet import COMETEnsemble
from src.training.ensemble.mbr import MBREnsemble
from src.training.ensemble.voting import VotingEnsemble

__all__ = [
    "AbstractEnsemble",
    "EnsembleResult",
    "EnsembleConfig",
    "COMETEnsemble",
    "MBREnsemble",
    "VotingEnsemble",
]
