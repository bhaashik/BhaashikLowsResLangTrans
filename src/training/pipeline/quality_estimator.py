"""
Quality estimation for translation outputs.

Provides fast, lightweight quality estimation to determine whether
translations from fine-tuned models are good enough or need API refinement.

Combines multiple signals:
- Model confidence (if available)
- Length ratio
- Character diversity
- N-gram repetition
- Optional: COMET-QE (quality estimation model)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import logging
import re

logger = logging.getLogger(__name__)


@dataclass
class QualityEstimate:
    """Quality estimate for a translation.

    Attributes:
        score: Overall quality score (0-1, higher is better)
        confidence: Model confidence if available
        length_ratio: Target/source length ratio
        diversity: Character diversity score
        repetition_penalty: Penalty for repeated n-grams
        signals: Individual signal scores
        needs_fallback: Whether translation should use fallback
    """
    score: float
    confidence: Optional[float] = None
    length_ratio: float = 1.0
    diversity: float = 1.0
    repetition_penalty: float = 0.0
    signals: Dict[str, float] = field(default_factory=dict)
    needs_fallback: bool = False


class QualityEstimator:
    """Estimate translation quality without references.

    Uses fast heuristics and optional neural models to estimate
    whether a translation is good enough or needs refinement.
    """

    def __init__(
        self,
        use_comet_qe: bool = False,
        comet_model: str = "Unbabel/wmt22-cometkiwi-da",
        threshold: float = 0.7,
        length_ratio_bounds: tuple = (0.5, 2.0),
        diversity_threshold: float = 0.2,
    ):
        """
        Initialize quality estimator.

        Args:
            use_comet_qe: Whether to use COMET-QE (slower but more accurate)
            comet_model: COMET-QE model for quality estimation
            threshold: Threshold for fallback decision (0-1)
            length_ratio_bounds: Acceptable length ratio range
            diversity_threshold: Minimum character diversity
        """
        self.use_comet_qe = use_comet_qe
        self.comet_model_name = comet_model
        self.threshold = threshold
        self.length_ratio_bounds = length_ratio_bounds
        self.diversity_threshold = diversity_threshold

        self.comet_model = None
        if self.use_comet_qe:
            self._load_comet_qe()

        logger.info(
            f"QualityEstimator initialized: use_comet_qe={use_comet_qe}, "
            f"threshold={threshold}"
        )

    def _load_comet_qe(self):
        """Load COMET-QE model for quality estimation."""
        try:
            from comet import download_model, load_from_checkpoint

            logger.info(f"Loading COMET-QE model: {self.comet_model_name}")
            model_path = download_model(self.comet_model_name)
            self.comet_model = load_from_checkpoint(model_path)
            self.comet_model.eval()

        except ImportError:
            logger.warning(
                "COMET not installed. Install with: pip install unbabel-comet"
            )
            self.use_comet_qe = False
        except Exception as e:
            logger.warning(f"Failed to load COMET-QE: {e}")
            self.use_comet_qe = False

    def estimate(
        self,
        source: str,
        translation: str,
        confidence: Optional[float] = None,
    ) -> QualityEstimate:
        """
        Estimate quality of a single translation.

        Args:
            source: Source text
            translation: Translation to evaluate
            confidence: Model confidence score if available

        Returns:
            QualityEstimate with score and signals
        """
        signals = {}

        # Signal 1: Model confidence (if available)
        if confidence is not None:
            signals['confidence'] = confidence
            confidence_weight = 0.4
        else:
            confidence_weight = 0.0

        # Signal 2: Length ratio
        length_ratio = len(translation) / (len(source) + 1)
        length_ok = (
            self.length_ratio_bounds[0] <= length_ratio <= self.length_ratio_bounds[1]
        )
        length_score = 1.0 if length_ok else max(
            0.0, 1.0 - abs(length_ratio - 1.0)
        )
        signals['length_ratio'] = length_score

        # Signal 3: Character diversity
        unique_chars = len(set(translation))
        total_chars = len(translation) + 1
        diversity = unique_chars / total_chars
        diversity_score = min(1.0, diversity / self.diversity_threshold)
        signals['diversity'] = diversity_score

        # Signal 4: Repetition detection
        repetition_penalty = self._detect_repetition(translation)
        signals['repetition'] = max(0.0, 1.0 - repetition_penalty)

        # Signal 5: COMET-QE (if enabled)
        if self.use_comet_qe and self.comet_model:
            comet_score = self._estimate_comet_qe(source, translation)
            signals['comet_qe'] = comet_score
            comet_weight = 0.4
        else:
            comet_score = 0.5
            comet_weight = 0.0

        # Compute weighted average
        if self.use_comet_qe and confidence is not None:
            # Use both confidence and COMET-QE
            score = (
                confidence_weight * confidence +
                comet_weight * comet_score +
                0.1 * length_score +
                0.05 * diversity_score +
                0.05 * signals['repetition']
            )
        elif confidence is not None:
            # Use confidence + heuristics
            score = (
                0.5 * confidence +
                0.2 * length_score +
                0.15 * diversity_score +
                0.15 * signals['repetition']
            )
        elif self.use_comet_qe:
            # Use COMET-QE + heuristics
            score = (
                0.6 * comet_score +
                0.15 * length_score +
                0.125 * diversity_score +
                0.125 * signals['repetition']
            )
        else:
            # Heuristics only
            score = (
                0.4 * length_score +
                0.3 * diversity_score +
                0.3 * signals['repetition']
            )

        # Clamp to [0, 1]
        score = max(0.0, min(1.0, score))

        # Decide if fallback needed
        needs_fallback = score < self.threshold

        return QualityEstimate(
            score=score,
            confidence=confidence,
            length_ratio=length_ratio,
            diversity=diversity,
            repetition_penalty=repetition_penalty,
            signals=signals,
            needs_fallback=needs_fallback,
        )

    def estimate_batch(
        self,
        sources: List[str],
        translations: List[str],
        confidences: Optional[List[float]] = None,
    ) -> List[QualityEstimate]:
        """
        Estimate quality for a batch of translations.

        Args:
            sources: Source texts
            translations: Translations to evaluate
            confidences: Model confidences if available

        Returns:
            List of QualityEstimate objects
        """
        if confidences is None:
            confidences = [None] * len(sources)

        estimates = []
        for source, translation, confidence in zip(
            sources, translations, confidences
        ):
            estimate = self.estimate(source, translation, confidence)
            estimates.append(estimate)

        return estimates

    def _detect_repetition(self, text: str) -> float:
        """
        Detect repeated n-grams (sign of poor quality).

        Args:
            text: Text to check

        Returns:
            Repetition penalty (0-1, higher means more repetition)
        """
        if len(text) < 10:
            return 0.0

        # Check trigram repetition
        words = text.split()
        if len(words) < 3:
            return 0.0

        trigrams = [
            ' '.join(words[i:i+3])
            for i in range(len(words) - 2)
        ]

        if not trigrams:
            return 0.0

        unique_trigrams = len(set(trigrams))
        total_trigrams = len(trigrams)

        repetition_ratio = 1.0 - (unique_trigrams / total_trigrams)

        # Penalize if > 30% repetition
        if repetition_ratio > 0.3:
            return min(1.0, repetition_ratio * 2)
        return 0.0

    def _estimate_comet_qe(self, source: str, translation: str) -> float:
        """
        Estimate quality using COMET-QE (reference-free).

        Args:
            source: Source text
            translation: Translation

        Returns:
            Quality score (0-1)
        """
        if not self.comet_model:
            return 0.5

        try:
            sample = {"src": source, "mt": translation}
            result = self.comet_model.predict([sample], batch_size=1, gpus=0)
            # COMET scores are typically in [-1, 1], normalize to [0, 1]
            score = (result.scores[0] + 1.0) / 2.0
            return max(0.0, min(1.0, score))
        except Exception as e:
            logger.warning(f"COMET-QE prediction failed: {e}")
            return 0.5


def compute_aggregate_quality(estimates: List[QualityEstimate]) -> Dict[str, float]:
    """
    Compute aggregate quality statistics.

    Args:
        estimates: List of quality estimates

    Returns:
        Dictionary with aggregate stats
    """
    if not estimates:
        return {}

    scores = [e.score for e in estimates]
    needs_fallback_count = sum(1 for e in estimates if e.needs_fallback)

    return {
        "mean_score": sum(scores) / len(scores),
        "min_score": min(scores),
        "max_score": max(scores),
        "fallback_percentage": 100.0 * needs_fallback_count / len(estimates),
        "total": len(estimates),
        "needs_fallback": needs_fallback_count,
    }
