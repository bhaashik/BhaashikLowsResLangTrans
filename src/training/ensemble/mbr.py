"""
Minimum Bayes Risk (MBR) decoding for ensemble translation.

MBR selects the translation that minimizes expected loss (or maximizes
expected utility) with respect to all candidate translations. Instead of
selecting the single most likely translation, MBR considers the entire
distribution of candidates.

The key insight is that the best translation is often not the highest
probability one, but the one that is most similar to the other good
candidates.

Algorithm:
    For each candidate c_i:
        utility(c_i) = Σ_j similarity(c_i, c_j)
    return argmax_i utility(c_i)

Reference:
    Kumar & Byrne (2004). Minimum Bayes-Risk Decoding for Statistical Machine Translation.
    Freitag et al. (2022). High Quality Rather than High Model Probability.
"""

from typing import List, Optional, Callable
import logging
import numpy as np

from src.training.ensemble.base import (
    AbstractEnsemble,
    EnsembleResult,
    EnsembleConfig,
    compute_pairwise_scores,
)

logger = logging.getLogger(__name__)


class MBREnsemble(AbstractEnsemble):
    """Ensemble method using Minimum Bayes Risk decoding.

    MBR selects the translation that has the highest expected similarity
    to all other candidate translations. This leverages the wisdom of the
    crowd - the best translation is one that many models agree on.

    Expected improvement: +0.8 to +1.5 BLEU over single best model
    """

    def __init__(
        self,
        models: List[Any],
        config: Optional[EnsembleConfig] = None,
        utility_metric: Optional[str] = None,
    ):
        """Initialize MBR ensemble.

        Args:
            models: List of translation models
            config: Ensemble configuration
            utility_metric: Utility metric ("bleu", "chrf", "comet")
        """
        super().__init__(models, config)

        self.utility_metric = utility_metric or self.config.mbr_utility
        self.metric_fn = self._get_metric_function(self.utility_metric)

        logger.info(f"MBR ensemble using utility metric: {self.utility_metric}")

    def _get_metric_function(self, metric_name: str) -> Callable:
        """Get the metric function for computing utility.

        Args:
            metric_name: Name of metric ("bleu", "chrf", "comet")

        Returns:
            Function that takes (hypothesis, reference) and returns score
        """
        if metric_name == "bleu":
            return self._compute_bleu_score
        elif metric_name == "chrf":
            return self._compute_chrf_score
        elif metric_name == "comet":
            return self._compute_comet_score
        else:
            raise ValueError(f"Unknown utility metric: {metric_name}")

    def translate(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        **kwargs
    ) -> EnsembleResult:
        """Translate texts using MBR decoding.

        Args:
            texts: Source texts to translate
            source_lang: Source language code
            target_lang: Target language code
            **kwargs: Additional arguments

        Returns:
            EnsembleResult with selected translations
        """
        logger.info(
            f"Translating {len(texts)} texts with MBR ensemble "
            f"(metric={self.utility_metric}, {len(self.models)} models)"
        )

        # Generate candidates from all models
        all_candidates = self.generate_candidates(
            texts, source_lang, target_lang
        )

        # Apply MBR to select best candidate
        translations = []
        selected_indices = []
        all_scores = []

        for text_idx, candidates in enumerate(all_candidates):
            logger.debug(
                f"Running MBR on {len(candidates)} candidates for text {text_idx+1}"
            )

            # Compute pairwise utilities
            utility_matrix = compute_pairwise_scores(
                candidates, self.metric_fn, normalize=True
            )

            # Compute expected utility for each candidate
            expected_utilities = [
                sum(utilities) / len(utilities)
                for utilities in utility_matrix
            ]

            # Select candidate with highest expected utility
            best_idx = max(
                range(len(expected_utilities)),
                key=lambda i: expected_utilities[i]
            )

            translations.append(candidates[best_idx])
            selected_indices.append(best_idx)
            all_scores.append(expected_utilities)

        # Create result
        result = EnsembleResult(
            translations=translations,
            candidates=all_candidates,
            scores=all_scores,
            selected_indices=selected_indices,
            method="mbr",
            metadata={
                "utility_metric": self.utility_metric,
                "num_models": len(self.models),
                "num_candidates": self.config.num_candidates,
                "temperature": self.config.temperature,
            }
        )

        logger.info("MBR ensemble translation completed")
        return result

    def _compute_bleu_score(self, hypothesis: str, reference: str) -> float:
        """Compute sentence-level BLEU score.

        Args:
            hypothesis: Candidate translation
            reference: Reference translation (or another candidate)

        Returns:
            BLEU score (0-100)
        """
        try:
            from sacrebleu.metrics import BLEU

            bleu = BLEU(effective_order=True)
            score = bleu.sentence_score(hypothesis, [reference])
            return score.score

        except ImportError:
            logger.error("sacrebleu not installed. Install with: pip install sacrebleu")
            raise

    def _compute_chrf_score(self, hypothesis: str, reference: str) -> float:
        """Compute chrF score.

        Args:
            hypothesis: Candidate translation
            reference: Reference translation (or another candidate)

        Returns:
            chrF score (0-100)
        """
        try:
            from sacrebleu.metrics import CHRF

            chrf = CHRF()
            score = chrf.sentence_score(hypothesis, [reference])
            return score.score

        except ImportError:
            logger.error("sacrebleu not installed. Install with: pip install sacrebleu")
            raise

    def _compute_comet_score(self, hypothesis: str, reference: str) -> float:
        """Compute COMET score (requires source text).

        Note: This is a simplified version. For full COMET scoring,
        use COMETEnsemble instead.

        Args:
            hypothesis: Candidate translation
            reference: Reference translation

        Returns:
            COMET score
        """
        # For MBR with COMET, we use reference-based COMET
        # This is less accurate than having source, but works for pairwise comparison
        logger.warning(
            "Using reference-based COMET for MBR. "
            "Consider using COMETEnsemble for better quality."
        )

        try:
            from comet import download_model, load_from_checkpoint

            if not hasattr(self, '_comet_model'):
                model_path = download_model(self.config.comet_model)
                self._comet_model = load_from_checkpoint(model_path)
                self._comet_model.eval()

            # Score without source (reference-free mode)
            sample = {"mt": hypothesis, "ref": reference}
            result = self._comet_model.predict([sample], batch_size=1, gpus=0)
            return result.scores[0]

        except ImportError:
            logger.error("COMET not installed. Install with: pip install unbabel-comet")
            raise


def mbr_decode_with_references(
    candidates: List[str],
    references: List[str],
    metric: str = "chrf"
) -> str:
    """MBR decoding with reference translations.

    This variant uses reference translations in addition to candidates
    when computing expected utility.

    Args:
        candidates: Candidate translations
        references: Reference translations
        metric: Utility metric to use

    Returns:
        Selected translation
    """
    # Create utility function
    if metric == "bleu":
        from sacrebleu.metrics import BLEU
        scorer = BLEU(effective_order=True)
        metric_fn = lambda h, r: scorer.sentence_score(h, [r]).score
    elif metric == "chrf":
        from sacrebleu.metrics import CHRF
        scorer = CHRF()
        metric_fn = lambda h, r: scorer.sentence_score(h, [r]).score
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Compute utility against references
    utilities = []
    for candidate in candidates:
        # Utility is average score against all references
        scores = [metric_fn(candidate, ref) for ref in references]
        utilities.append(sum(scores) / len(scores))

    # Select candidate with highest utility
    best_idx = max(range(len(utilities)), key=lambda i: utilities[i])
    return candidates[best_idx]


def quality_aware_mbr(
    candidates: List[str],
    quality_scores: List[float],
    metric_fn: Callable,
    quality_weight: float = 0.5
) -> str:
    """MBR decoding with quality-weighted utility.

    Combines MBR utility with external quality estimates (e.g., from COMET).

    Args:
        candidates: Candidate translations
        quality_scores: Quality scores for each candidate
        metric_fn: Function to compute pairwise similarity
        quality_weight: Weight for quality scores (0-1)

    Returns:
        Selected translation
    """
    # Compute pairwise utilities
    utility_matrix = compute_pairwise_scores(candidates, metric_fn, normalize=True)

    # Compute expected utility
    expected_utilities = [
        sum(utilities) / len(utilities)
        for utilities in utility_matrix
    ]

    # Normalize quality scores
    quality_scores = np.array(quality_scores)
    quality_scores = (quality_scores - quality_scores.min()) / (
        quality_scores.max() - quality_scores.min() + 1e-8
    )

    # Combine utilities and quality scores
    combined_scores = [
        (1 - quality_weight) * utility + quality_weight * quality
        for utility, quality in zip(expected_utilities, quality_scores)
    ]

    # Select best
    best_idx = max(range(len(combined_scores)), key=lambda i: combined_scores[i])
    return candidates[best_idx]
