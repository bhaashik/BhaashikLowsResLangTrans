"""SMT-inspired quality components for MT training.

This module provides quality assessment components inspired by Statistical MT:
- Adequacy: Measures semantic similarity between source and translation
- Fluency: Measures syntactic quality of translation

These components can be used for:
1. Quality-aware training (weighted loss)
2. Quality estimation
3. Automatic post-editing
4. Model selection

References:
- Specia et al. (2010): Translation Quality Estimation
- Fomicheva et al. (2020): Unsupervised Quality Estimation
- Rei et al. (2020): COMET - Cross-lingual Optimized Metric for Evaluation
"""

from src.training.quality.config import QualityConfig
from src.training.quality.adequacy import (
    AbstractAdequacyEstimator,
    SentenceEmbeddingAdequacy,
    EntailmentAdequacy,
)
from src.training.quality.fluency import (
    AbstractFluencyEstimator,
    PerplexityFluency,
    ParseBasedFluency,
)
from src.training.quality.quality_estimator import QualityEstimator
from src.training.quality.losses import (
    QualityWeightedLoss,
    AdequacyLoss,
    FluencyLoss,
)

__all__ = [
    "QualityConfig",
    "AbstractAdequacyEstimator",
    "SentenceEmbeddingAdequacy",
    "EntailmentAdequacy",
    "AbstractFluencyEstimator",
    "PerplexityFluency",
    "ParseBasedFluency",
    "QualityEstimator",
    "QualityWeightedLoss",
    "AdequacyLoss",
    "FluencyLoss",
]
