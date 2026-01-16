"""
Base classes and utilities for ensemble translation methods.

Provides abstract interface and common functionality for combining
translations from multiple models.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class EnsembleConfig:
    """Configuration for ensemble methods.

    Attributes:
        method: Ensemble method name ("comet", "mbr", "voting", etc.)
        num_candidates: Number of candidate translations per model
        temperature: Sampling temperature for candidate generation
        comet_model: COMET model for quality estimation
        mbr_utility: Utility metric for MBR ("bleu", "chrf", "comet")
        voting_strategy: Strategy for voting ("majority", "weighted", "ranked")
        weights: Model weights for weighted voting
        use_cache: Whether to cache quality scores
        batch_size: Batch size for quality estimation
        device: Device for running ensemble (cuda/cpu)
        custom_params: Additional custom parameters
    """
    method: str = "comet"
    num_candidates: int = 5
    temperature: float = 0.8

    # COMET-specific
    comet_model: str = "Unbabel/wmt22-comet-da"

    # MBR-specific
    mbr_utility: str = "chrf"  # bleu, chrf, comet

    # Voting-specific
    voting_strategy: str = "majority"  # majority, weighted, ranked
    weights: Optional[List[float]] = None

    # Performance
    use_cache: bool = True
    batch_size: int = 32
    device: str = "cuda"

    # Extensibility
    custom_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "method": self.method,
            "num_candidates": self.num_candidates,
            "temperature": self.temperature,
            "comet_model": self.comet_model,
            "mbr_utility": self.mbr_utility,
            "voting_strategy": self.voting_strategy,
            "weights": self.weights,
            "use_cache": self.use_cache,
            "batch_size": self.batch_size,
            "device": self.device,
            **self.custom_params,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "EnsembleConfig":
        """Create config from dictionary."""
        known_fields = {
            "method", "num_candidates", "temperature", "comet_model",
            "mbr_utility", "voting_strategy", "weights", "use_cache",
            "batch_size", "device"
        }
        custom_params = {k: v for k, v in config_dict.items() if k not in known_fields}
        known_params = {k: v for k, v in config_dict.items() if k in known_fields}
        return cls(**known_params, custom_params=custom_params)


@dataclass
class EnsembleResult:
    """Result from ensemble translation.

    Attributes:
        translations: Final selected translations
        candidates: All candidate translations from each model
        scores: Quality scores for each candidate
        selected_indices: Indices of selected candidates
        method: Ensemble method used
        metadata: Additional metadata (timing, model info, etc.)
    """
    translations: List[str]
    candidates: List[List[str]]  # [num_sources][num_models * num_candidates]
    scores: List[List[float]]  # [num_sources][num_models * num_candidates]
    selected_indices: List[int]  # Index of selected candidate for each source
    method: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_best_candidates(self, top_k: int = 3) -> List[List[Tuple[str, float]]]:
        """Get top-k candidates with scores for each source.

        Args:
            top_k: Number of top candidates to return

        Returns:
            List of (translation, score) tuples for each source
        """
        result = []
        for candidates, scores in zip(self.candidates, self.scores):
            # Sort by score (descending)
            sorted_pairs = sorted(
                zip(candidates, scores),
                key=lambda x: x[1],
                reverse=True
            )
            result.append(sorted_pairs[:top_k])
        return result


class AbstractEnsemble(ABC):
    """Abstract base class for ensemble translation methods.

    All ensemble methods should inherit from this class and implement
    the translate() method. The class provides common utilities for
    generating candidates and computing scores.
    """

    def __init__(
        self,
        models: List[Any],
        config: Optional[EnsembleConfig] = None,
    ):
        """Initialize ensemble.

        Args:
            models: List of translation models (AbstractMTModel instances)
            config: Ensemble configuration
        """
        self.models = models
        self.config = config or EnsembleConfig()
        self._score_cache = {} if self.config.use_cache else None

        logger.info(
            f"Initialized {self.__class__.__name__} with {len(models)} models, "
            f"method={self.config.method}"
        )

    @abstractmethod
    def translate(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        **kwargs
    ) -> EnsembleResult:
        """Translate texts using ensemble method.

        Args:
            texts: Source texts to translate
            source_lang: Source language code
            target_lang: Target language code
            **kwargs: Additional arguments

        Returns:
            EnsembleResult with translations and metadata
        """
        pass

    def generate_candidates(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        num_candidates: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> List[List[str]]:
        """Generate candidate translations from all models.

        Args:
            texts: Source texts
            source_lang: Source language code
            target_lang: Target language code
            num_candidates: Number of candidates per model (overrides config)
            temperature: Sampling temperature (overrides config)

        Returns:
            List of candidates for each source text
            Shape: [num_sources][num_models * num_candidates]
        """
        num_candidates = num_candidates or self.config.num_candidates
        temperature = temperature or self.config.temperature

        all_candidates = [[] for _ in texts]

        for model_idx, model in enumerate(self.models):
            logger.debug(f"Generating candidates from model {model_idx+1}/{len(self.models)}")

            # Generate multiple candidates per text
            for text_idx, text in enumerate(texts):
                candidates = model.translate(
                    [text],
                    source_lang=source_lang,
                    target_lang=target_lang,
                    num_return_sequences=num_candidates,
                    temperature=temperature,
                    do_sample=True if num_candidates > 1 else False,
                )
                all_candidates[text_idx].extend(candidates)

        return all_candidates

    def _get_cache_key(
        self,
        source: str,
        candidate: str,
        metric: str
    ) -> str:
        """Generate cache key for score."""
        return f"{metric}:{hash(source)}:{hash(candidate)}"

    def _get_cached_score(
        self,
        source: str,
        candidate: str,
        metric: str
    ) -> Optional[float]:
        """Get cached score if available."""
        if self._score_cache is None:
            return None
        key = self._get_cache_key(source, candidate, metric)
        return self._score_cache.get(key)

    def _cache_score(
        self,
        source: str,
        candidate: str,
        metric: str,
        score: float
    ):
        """Cache a score."""
        if self._score_cache is not None:
            key = self._get_cache_key(source, candidate, metric)
            self._score_cache[key] = score


def compute_pairwise_scores(
    candidates: List[str],
    metric_fn: callable,
    normalize: bool = True
) -> List[List[float]]:
    """Compute pairwise scores between candidates.

    Used for MBR to compute utility of each candidate against all others.

    Args:
        candidates: List of candidate translations
        metric_fn: Function that takes (hypothesis, reference) and returns score
        normalize: Whether to normalize scores to [0, 1]

    Returns:
        Pairwise score matrix [num_candidates][num_candidates]
    """
    n = len(candidates)
    scores = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                scores[i][j] = 1.0 if normalize else 100.0
            else:
                score = metric_fn(candidates[i], candidates[j])
                scores[i][j] = score

    # Normalize if requested
    if normalize:
        max_score = max(max(row) for row in scores)
        min_score = min(min(row) for row in scores)
        if max_score > min_score:
            scores = [
                [(s - min_score) / (max_score - min_score) for s in row]
                for row in scores
            ]

    return scores


def select_by_scores(
    candidates: List[str],
    scores: List[float],
    strategy: str = "max"
) -> Tuple[str, int, float]:
    """Select best candidate based on scores.

    Args:
        candidates: Candidate translations
        scores: Scores for each candidate
        strategy: Selection strategy ("max", "min", "mean_max")

    Returns:
        (selected_translation, selected_index, score)
    """
    if strategy == "max":
        idx = max(range(len(scores)), key=lambda i: scores[i])
    elif strategy == "min":
        idx = min(range(len(scores)), key=lambda i: scores[i])
    elif strategy == "mean_max":
        # Select candidate with highest mean score against others
        mean_scores = [sum(s) / len(s) for s in scores]
        idx = max(range(len(mean_scores)), key=lambda i: mean_scores[i])
    else:
        raise ValueError(f"Unknown selection strategy: {strategy}")

    return candidates[idx], idx, scores[idx]
