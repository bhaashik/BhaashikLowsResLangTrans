"""
Hybrid translation pipeline for quality-based routing.

This module implements a hybrid translation strategy that combines:
1. Fine-tuned models (primary, low-cost)
2. API-based translation (fallback for low-confidence translations)
3. Ensemble methods (for critical/high-stakes translations)

The pipeline routes translations based on configurable quality thresholds,
achieving optimal quality/cost tradeoffs.

Usage:
    from src.training.pipeline import HybridPipeline, HybridConfig

    config = HybridConfig(
        finetuned_percentage=90.0,
        api_percentage=9.0,
        ensemble_percentage=1.0
    )

    pipeline = HybridPipeline(
        finetuned_model=finetuned_provider,
        api_model=openai_provider,
        config=config
    )

    results = pipeline.translate(texts, source_lang="hi", target_lang="bho")
"""

from src.training.pipeline.hybrid import (
    HybridPipeline,
    HybridConfig,
    HybridResult,
    RouteDecision,
)

from src.training.pipeline.quality_estimator import (
    QualityEstimator,
    QualityEstimate,
)

__all__ = [
    "HybridPipeline",
    "HybridConfig",
    "HybridResult",
    "RouteDecision",
    "QualityEstimator",
    "QualityEstimate",
]
