"""
Voting-based ensemble methods for translation.

Implements various voting strategies to combine translations from multiple
models:

1. Majority Voting: Select the most common translation
2. Weighted Voting: Vote weighted by model quality/confidence
3. Ranked Voting: Aggregate rankings from multiple metrics

Voting works best when:
- Models produce similar but not identical translations
- There's a clear consensus among good translations
- You want a simple, interpretable ensemble method
"""

from typing import List, Optional, Dict, Any, Tuple
from collections import Counter
import logging

from src.training.ensemble.base import (
    AbstractEnsemble,
    EnsembleResult,
    EnsembleConfig,
)

logger = logging.getLogger(__name__)


class VotingEnsemble(AbstractEnsemble):
    """Ensemble method using voting strategies.

    Supports multiple voting strategies:
    - Majority: Most common translation wins
    - Weighted: Models vote with different weights
    - Ranked: Combine rankings from quality metrics

    Expected improvement: +0.3 to +0.8 BLEU over single best model
    """

    def __init__(
        self,
        models: List[Any],
        config: Optional[EnsembleConfig] = None,
        model_weights: Optional[List[float]] = None,
    ):
        """Initialize voting ensemble.

        Args:
            models: List of translation models
            config: Ensemble configuration
            model_weights: Weights for each model (for weighted voting)
        """
        super().__init__(models, config)

        self.strategy = self.config.voting_strategy
        self.model_weights = model_weights or self.config.weights

        # Default to uniform weights
        if self.model_weights is None:
            self.model_weights = [1.0] * len(self.models)

        # Normalize weights
        total = sum(self.model_weights)
        self.model_weights = [w / total for w in self.model_weights]

        logger.info(
            f"Voting ensemble with strategy={self.strategy}, "
            f"weights={self.model_weights}"
        )

    def translate(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        **kwargs
    ) -> EnsembleResult:
        """Translate texts using voting.

        Args:
            texts: Source texts to translate
            source_lang: Source language code
            target_lang: Target language code
            **kwargs: Additional arguments

        Returns:
            EnsembleResult with selected translations
        """
        logger.info(
            f"Translating {len(texts)} texts with voting ensemble "
            f"(strategy={self.strategy}, {len(self.models)} models)"
        )

        # Get single best translation from each model
        # (Not using num_candidates for voting - each model votes once)
        all_translations = []
        for model_idx, model in enumerate(self.models):
            translations = model.translate(
                texts,
                source_lang=source_lang,
                target_lang=target_lang,
                num_return_sequences=1,
            )
            all_translations.append(translations)

        # Apply voting strategy
        if self.strategy == "majority":
            results = self._majority_voting(texts, all_translations)
        elif self.strategy == "weighted":
            results = self._weighted_voting(texts, all_translations)
        elif self.strategy == "ranked":
            results = self._ranked_voting(texts, all_translations)
        else:
            raise ValueError(f"Unknown voting strategy: {self.strategy}")

        translations, selected_indices, scores, candidates = results

        # Create result
        result = EnsembleResult(
            translations=translations,
            candidates=candidates,
            scores=scores,
            selected_indices=selected_indices,
            method=f"voting_{self.strategy}",
            metadata={
                "strategy": self.strategy,
                "num_models": len(self.models),
                "model_weights": self.model_weights,
            }
        )

        logger.info("Voting ensemble translation completed")
        return result

    def _majority_voting(
        self,
        sources: List[str],
        all_translations: List[List[str]]
    ) -> Tuple[List[str], List[int], List[List[float]], List[List[str]]]:
        """Majority voting: select most common translation.

        Args:
            sources: Source texts
            all_translations: Translations from each model [num_models][num_sources]

        Returns:
            (translations, selected_indices, scores, candidates)
        """
        translations = []
        selected_indices = []
        all_scores = []
        all_candidates = []

        num_sources = len(sources)
        num_models = len(all_translations)

        # Transpose to get translations per source
        for src_idx in range(num_sources):
            # Get all translations for this source
            candidates = [
                all_translations[model_idx][src_idx]
                for model_idx in range(num_models)
            ]

            # Count occurrences
            counts = Counter(candidates)
            most_common = counts.most_common()

            # Select most common
            selected = most_common[0][0]
            selected_count = most_common[0][1]

            # Find index of selected translation
            selected_idx = candidates.index(selected)

            # Compute vote scores (normalized counts)
            unique_candidates = list(set(candidates))
            scores = [
                counts[candidate] / num_models
                for candidate in unique_candidates
            ]

            translations.append(selected)
            selected_indices.append(selected_idx)
            all_scores.append(scores)
            all_candidates.append(unique_candidates)

        return translations, selected_indices, all_scores, all_candidates

    def _weighted_voting(
        self,
        sources: List[str],
        all_translations: List[List[str]]
    ) -> Tuple[List[str], List[int], List[List[float]], List[List[str]]]:
        """Weighted voting: models vote with different weights.

        Args:
            sources: Source texts
            all_translations: Translations from each model

        Returns:
            (translations, selected_indices, scores, candidates)
        """
        translations = []
        selected_indices = []
        all_scores = []
        all_candidates = []

        num_sources = len(sources)
        num_models = len(all_translations)

        for src_idx in range(num_sources):
            # Get all translations for this source with weights
            candidates = [
                all_translations[model_idx][src_idx]
                for model_idx in range(num_models)
            ]

            # Compute weighted votes
            weighted_counts = {}
            for candidate, weight in zip(candidates, self.model_weights):
                weighted_counts[candidate] = (
                    weighted_counts.get(candidate, 0.0) + weight
                )

            # Select candidate with highest weighted vote
            selected = max(weighted_counts, key=weighted_counts.get)
            selected_idx = candidates.index(selected)

            # Compute scores (normalized weighted votes)
            unique_candidates = list(weighted_counts.keys())
            total_weight = sum(weighted_counts.values())
            scores = [
                weighted_counts[candidate] / total_weight
                for candidate in unique_candidates
            ]

            translations.append(selected)
            selected_indices.append(selected_idx)
            all_scores.append(scores)
            all_candidates.append(unique_candidates)

        return translations, selected_indices, all_scores, all_candidates

    def _ranked_voting(
        self,
        sources: List[str],
        all_translations: List[List[str]]
    ) -> Tuple[List[str], List[int], List[List[float]], List[List[str]]]:
        """Ranked voting using quality metrics.

        Uses multiple quality metrics to rank candidates, then combines
        rankings to select the best translation.

        Args:
            sources: Source texts
            all_translations: Translations from each model

        Returns:
            (translations, selected_indices, scores, candidates)
        """
        # For ranked voting, we need to score candidates with metrics
        # We'll use length ratio and character n-gram diversity as simple metrics

        translations = []
        selected_indices = []
        all_scores = []
        all_candidates = []

        num_sources = len(sources)
        num_models = len(all_translations)

        for src_idx in range(num_sources):
            source = sources[src_idx]
            candidates = [
                all_translations[model_idx][src_idx]
                for model_idx in range(num_models)
            ]

            # Score each candidate with multiple metrics
            candidate_scores = {}
            for candidate in set(candidates):
                # Metric 1: Length ratio (closer to 1.0 is better)
                length_ratio = len(candidate) / (len(source) + 1)
                length_score = 1.0 / (1.0 + abs(1.0 - length_ratio))

                # Metric 2: Character diversity
                diversity = len(set(candidate)) / (len(candidate) + 1)

                # Metric 3: Vote count (how many models produced this)
                vote_count = candidates.count(candidate) / num_models

                # Combined score (can be weighted)
                combined_score = (
                    0.3 * length_score +
                    0.2 * diversity +
                    0.5 * vote_count
                )
                candidate_scores[candidate] = combined_score

            # Select candidate with highest combined score
            selected = max(candidate_scores, key=candidate_scores.get)
            selected_idx = candidates.index(selected)

            # Get unique candidates and their scores
            unique_candidates = list(candidate_scores.keys())
            scores = [candidate_scores[c] for c in unique_candidates]

            translations.append(selected)
            selected_indices.append(selected_idx)
            all_scores.append(scores)
            all_candidates.append(unique_candidates)

        return translations, selected_indices, all_scores, all_candidates


def confidence_weighted_voting(
    translations: List[str],
    confidences: List[float],
    threshold: float = 0.5
) -> str:
    """Voting weighted by model confidence.

    Args:
        translations: Translations from different models
        confidences: Confidence scores for each translation
        threshold: Minimum confidence to participate in voting

    Returns:
        Selected translation
    """
    # Filter by confidence threshold
    filtered = [
        (trans, conf)
        for trans, conf in zip(translations, confidences)
        if conf >= threshold
    ]

    if not filtered:
        # If no translation meets threshold, use highest confidence
        best_idx = max(range(len(confidences)), key=lambda i: confidences[i])
        return translations[best_idx]

    # Weighted voting
    weighted_counts = {}
    for trans, conf in filtered:
        weighted_counts[trans] = weighted_counts.get(trans, 0.0) + conf

    return max(weighted_counts, key=weighted_counts.get)


def borda_count_voting(
    candidates: List[str],
    rankings: List[List[str]]
) -> str:
    """Borda count voting from multiple rankings.

    Each model provides a ranking of candidates. Borda count assigns
    points based on position in ranking (higher position = more points).

    Args:
        candidates: List of unique candidate translations
        rankings: List of rankings from different models/metrics
                  Each ranking is a list of candidates in preference order

    Returns:
        Selected translation
    """
    # Compute Borda scores
    borda_scores = {candidate: 0 for candidate in candidates}

    for ranking in rankings:
        n = len(ranking)
        for position, candidate in enumerate(ranking):
            # Award points: n-position (first place gets n points, last gets 1)
            borda_scores[candidate] += n - position

    return max(borda_scores, key=borda_scores.get)
