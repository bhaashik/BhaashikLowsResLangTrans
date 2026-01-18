"""
Quality estimator combining adequacy and fluency.

Provides unified quality assessment by combining adequacy (semantic similarity)
and fluency (syntactic quality) components.
"""

from typing import List, Optional, Dict, Tuple
import numpy as np
import logging

from src.training.quality.config import QualityConfig
from src.training.quality.adequacy import (
    AbstractAdequacyEstimator,
    create_adequacy_estimator,
)
from src.training.quality.fluency import (
    AbstractFluencyEstimator,
    create_fluency_estimator,
)

logger = logging.getLogger(__name__)


class QualityEstimator:
    """
    Quality estimator combining adequacy and fluency.

    Provides overall quality scores by combining:
    - Adequacy: How well the translation preserves meaning
    - Fluency: How natural/fluent the translation is
    """

    def __init__(
        self,
        config: QualityConfig,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
    ):
        """
        Initialize quality estimator.

        Args:
            config: Quality configuration
            source_lang: Source language code (for language-specific models)
            target_lang: Target language code (for fluency estimation)
        """
        self.config = config
        self.source_lang = source_lang
        self.target_lang = target_lang

        self.adequacy_estimator: Optional[AbstractAdequacyEstimator] = None
        self.fluency_estimator: Optional[AbstractFluencyEstimator] = None

        self._initialized = False

    def initialize(self):
        """Initialize adequacy and fluency estimators."""
        if self._initialized:
            return

        logger.info("Initializing QualityEstimator...")

        # Initialize adequacy estimator
        if self.config.use_adequacy:
            logger.info(f"  Creating adequacy estimator ({self.config.adequacy_method})...")
            self.adequacy_estimator = create_adequacy_estimator(
                method=self.config.adequacy_method,
                model_name=self.config.adequacy_model,
                device=self.config.device,
                batch_size=self.config.batch_size,
                cache=self.config.cache_quality_scores,
            )
            logger.info("  ✓ Adequacy estimator ready")

        # Initialize fluency estimator
        if self.config.use_fluency:
            logger.info(f"  Creating fluency estimator ({self.config.fluency_method})...")
            self.fluency_estimator = create_fluency_estimator(
                method=self.config.fluency_method,
                model_name=self.config.fluency_model,
                language=self.target_lang,
                device=self.config.device,
                batch_size=self.config.batch_size,
                cache=self.config.cache_quality_scores,
            )
            logger.info("  ✓ Fluency estimator ready")

        self._initialized = True
        logger.info("✓ QualityEstimator initialized")

    def estimate_quality(
        self,
        source_texts: List[str],
        translation_texts: List[str],
        return_components: bool = False,
    ) -> np.ndarray:
        """
        Estimate quality scores for translations.

        Args:
            source_texts: Source sentences
            translation_texts: Translation sentences
            return_components: If True, return (quality, adequacy, fluency)

        Returns:
            Quality scores [batch_size] or tuple of (quality, adequacy, fluency)
        """
        if not self._initialized:
            self.initialize()

        batch_size = len(source_texts)
        adequacy_scores = np.zeros(batch_size)
        fluency_scores = np.zeros(batch_size)

        # Estimate adequacy
        if self.config.use_adequacy:
            adequacy_scores = self.adequacy_estimator.estimate_adequacy(
                source_texts,
                translation_texts
            )

        # Estimate fluency
        if self.config.use_fluency:
            fluency_scores = self.fluency_estimator.estimate_fluency(
                translation_texts
            )

        # Combine scores
        quality_scores = (
            self.config.adequacy_weight * adequacy_scores +
            self.config.fluency_weight * fluency_scores
        )

        # Normalize if requested
        if self.config.normalize_scores:
            quality_scores = np.clip(quality_scores, 0.0, 1.0)

        if return_components:
            return quality_scores, adequacy_scores, fluency_scores
        else:
            return quality_scores

    def estimate_single(
        self,
        source_text: str,
        translation_text: str,
        return_components: bool = False,
    ) -> float:
        """
        Estimate quality for single translation.

        Args:
            source_text: Source sentence
            translation_text: Translation sentence
            return_components: If True, return (quality, adequacy, fluency)

        Returns:
            Quality score or tuple of (quality, adequacy, fluency)
        """
        if return_components:
            quality, adequacy, fluency = self.estimate_quality(
                [source_text],
                [translation_text],
                return_components=True
            )
            return float(quality[0]), float(adequacy[0]), float(fluency[0])
        else:
            quality = self.estimate_quality(
                [source_text],
                [translation_text],
                return_components=False
            )
            return float(quality[0])

    def filter_by_quality(
        self,
        source_texts: List[str],
        translation_texts: List[str],
        threshold: Optional[float] = None,
    ) -> Tuple[List[str], List[str], np.ndarray]:
        """
        Filter translations by quality threshold.

        Args:
            source_texts: Source sentences
            translation_texts: Translation sentences
            threshold: Quality threshold (uses config.quality_threshold if None)

        Returns:
            Tuple of (filtered_sources, filtered_translations, quality_scores)
        """
        threshold = threshold or self.config.quality_threshold

        # Estimate quality
        quality_scores = self.estimate_quality(source_texts, translation_texts)

        # Filter
        mask = quality_scores >= threshold
        filtered_sources = [src for src, m in zip(source_texts, mask) if m]
        filtered_translations = [tgt for tgt, m in zip(translation_texts, mask) if m]

        logger.info(
            f"Filtered {len(source_texts)} → {len(filtered_sources)} "
            f"(threshold={threshold:.3f})"
        )

        return filtered_sources, filtered_translations, quality_scores

    def get_statistics(
        self,
        source_texts: List[str],
        translation_texts: List[str],
    ) -> Dict[str, float]:
        """
        Get quality statistics for a dataset.

        Args:
            source_texts: Source sentences
            translation_texts: Translation sentences

        Returns:
            Dictionary with statistics
        """
        quality, adequacy, fluency = self.estimate_quality(
            source_texts,
            translation_texts,
            return_components=True
        )

        stats = {
            "quality_mean": float(np.mean(quality)),
            "quality_std": float(np.std(quality)),
            "quality_min": float(np.min(quality)),
            "quality_max": float(np.max(quality)),
            "quality_median": float(np.median(quality)),
        }

        if self.config.use_adequacy:
            stats.update({
                "adequacy_mean": float(np.mean(adequacy)),
                "adequacy_std": float(np.std(adequacy)),
            })

        if self.config.use_fluency:
            stats.update({
                "fluency_mean": float(np.mean(fluency)),
                "fluency_std": float(np.std(fluency)),
            })

        return stats

    def cleanup(self):
        """Cleanup resources."""
        if self.adequacy_estimator:
            self.adequacy_estimator.cleanup()
        if self.fluency_estimator:
            self.fluency_estimator.cleanup()


def create_quality_estimator(
    config: QualityConfig,
    source_lang: Optional[str] = None,
    target_lang: Optional[str] = None,
) -> QualityEstimator:
    """
    Create and initialize quality estimator.

    Args:
        config: Quality configuration
        source_lang: Source language code
        target_lang: Target language code

    Returns:
        Initialized QualityEstimator
    """
    estimator = QualityEstimator(config, source_lang, target_lang)
    estimator.initialize()
    return estimator
