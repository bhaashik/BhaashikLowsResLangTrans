"""
Quality-aware loss functions.

Provides loss functions that incorporate quality estimation:
- Quality-weighted loss: Weight examples by quality
- Adequacy loss: Semantic similarity loss
- Fluency loss: Target-side fluency loss
"""

from typing import List, Optional, Dict
import torch
import torch.nn as nn
import numpy as np
import logging

from src.training.quality.quality_estimator import QualityEstimator
from src.training.quality.config import QualityConfig

logger = logging.getLogger(__name__)


class QualityWeightedLoss(nn.Module):
    """
    Quality-weighted cross-entropy loss.

    Weights each example by its quality score to focus learning
    on high-quality examples.
    """

    def __init__(
        self,
        quality_estimator: QualityEstimator,
        base_loss: Optional[nn.Module] = None,
        weighting_scheme: str = "linear",
        temperature: float = 1.0,
        min_weight: float = 0.1,
    ):
        """
        Initialize quality-weighted loss.

        Args:
            quality_estimator: Quality estimator for computing weights
            base_loss: Base loss function (uses CrossEntropyLoss if None)
            weighting_scheme: How to convert quality to weights
                - linear: weight = quality
                - squared: weight = quality^2
                - exponential: weight = exp(quality / temperature)
            temperature: Temperature for exponential weighting
            min_weight: Minimum weight (prevent zero weights)
        """
        super().__init__()
        self.quality_estimator = quality_estimator
        self.base_loss = base_loss or nn.CrossEntropyLoss(reduction='none')
        self.weighting_scheme = weighting_scheme
        self.temperature = temperature
        self.min_weight = min_weight

        # Cache for quality scores
        self._quality_cache: Dict[str, float] = {}

    def compute_weights(
        self,
        source_texts: List[str],
        translation_texts: List[str],
        use_cache: bool = True,
    ) -> torch.Tensor:
        """
        Compute quality-based weights for examples.

        Args:
            source_texts: Source sentences
            translation_texts: Translation sentences
            use_cache: Whether to use cached quality scores

        Returns:
            Weights tensor [batch_size]
        """
        # Get quality scores
        quality_scores = []
        for src, tgt in zip(source_texts, translation_texts):
            cache_key = f"{src}|||{tgt}"

            if use_cache and cache_key in self._quality_cache:
                score = self._quality_cache[cache_key]
            else:
                score = self.quality_estimator.estimate_single(src, tgt)
                if use_cache:
                    self._quality_cache[cache_key] = score

            quality_scores.append(score)

        quality_scores = np.array(quality_scores)

        # Convert quality to weights
        if self.weighting_scheme == "linear":
            weights = quality_scores
        elif self.weighting_scheme == "squared":
            weights = quality_scores ** 2
        elif self.weighting_scheme == "exponential":
            weights = np.exp(quality_scores / self.temperature)
        else:
            raise ValueError(f"Unknown weighting scheme: {self.weighting_scheme}")

        # Apply minimum weight
        weights = np.maximum(weights, self.min_weight)

        # Normalize weights to sum to batch_size (preserve scale)
        weights = weights / weights.mean()

        return torch.from_numpy(weights).float()

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        source_texts: Optional[List[str]] = None,
        translation_texts: Optional[List[str]] = None,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute quality-weighted loss.

        Args:
            logits: Model logits [batch, seq_len, vocab]
            labels: Target labels [batch, seq_len]
            source_texts: Source sentences (for quality estimation)
            translation_texts: Translation sentences (for quality estimation)
            weights: Pre-computed weights [batch] (computes if None)

        Returns:
            Weighted loss scalar
        """
        # Compute base loss
        if hasattr(self.base_loss, 'ignore_index'):
            loss = self.base_loss(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
        else:
            loss = self.base_loss(logits, labels)

        # Reshape to [batch, seq_len]
        if loss.dim() == 1:
            loss = loss.view(labels.size())

        # Compute per-example loss
        loss_per_example = loss.sum(dim=1)  # [batch]

        # Get or compute weights
        if weights is None:
            if source_texts is None or translation_texts is None:
                raise ValueError(
                    "Must provide either weights or (source_texts, translation_texts)"
                )
            weights = self.compute_weights(source_texts, translation_texts)

        # Move weights to same device as loss
        weights = weights.to(loss.device)

        # Apply weights
        weighted_loss = loss_per_example * weights

        # Return mean
        return weighted_loss.mean()


class AdequacyLoss(nn.Module):
    """
    Adequacy loss using semantic similarity.

    Encourages model to generate translations with high semantic
    similarity to source.
    """

    def __init__(
        self,
        quality_estimator: QualityEstimator,
        margin: float = 0.2,
    ):
        """
        Initialize adequacy loss.

        Args:
            quality_estimator: Quality estimator with adequacy component
            margin: Margin for ranking loss
        """
        super().__init__()
        self.quality_estimator = quality_estimator

        if not quality_estimator.config.use_adequacy:
            raise ValueError("Quality estimator must have adequacy enabled")

        self.margin = margin

    def forward(
        self,
        source_texts: List[str],
        positive_texts: List[str],
        negative_texts: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Compute adequacy loss.

        Args:
            source_texts: Source sentences
            positive_texts: Good translations (high adequacy)
            negative_texts: Bad translations (low adequacy, optional)

        Returns:
            Adequacy loss
        """
        # Get adequacy for positive examples
        _, adequacy_pos, _ = self.quality_estimator.estimate_quality(
            source_texts,
            positive_texts,
            return_components=True
        )

        if negative_texts is not None:
            # Ranking loss: adequacy(pos) > adequacy(neg) + margin
            _, adequacy_neg, _ = self.quality_estimator.estimate_quality(
                source_texts,
                negative_texts,
                return_components=True
            )

            # Margin ranking loss
            loss = torch.clamp(
                self.margin + torch.from_numpy(adequacy_neg) - torch.from_numpy(adequacy_pos),
                min=0.0
            ).mean()
        else:
            # Maximize adequacy (minimize negative adequacy)
            loss = -torch.from_numpy(adequacy_pos).mean()

        return loss


class FluencyLoss(nn.Module):
    """
    Fluency loss using target-side quality.

    Encourages model to generate fluent translations.
    """

    def __init__(
        self,
        quality_estimator: QualityEstimator,
    ):
        """
        Initialize fluency loss.

        Args:
            quality_estimator: Quality estimator with fluency component
        """
        super().__init__()
        self.quality_estimator = quality_estimator

        if not quality_estimator.config.use_fluency:
            raise ValueError("Quality estimator must have fluency enabled")

    def forward(
        self,
        translation_texts: List[str],
    ) -> torch.Tensor:
        """
        Compute fluency loss.

        Args:
            translation_texts: Translations to score

        Returns:
            Fluency loss (negative fluency to minimize)
        """
        # Get fluency scores
        _, _, fluency = self.quality_estimator.estimate_quality(
            [""] * len(translation_texts),  # Dummy sources (not used for fluency)
            translation_texts,
            return_components=True
        )

        # Minimize negative fluency (maximize fluency)
        loss = -torch.from_numpy(fluency).mean()

        return loss


class CombinedQualityLoss(nn.Module):
    """
    Combined quality loss: base loss + adequacy + fluency.

    Provides multi-objective training with quality components.
    """

    def __init__(
        self,
        quality_estimator: QualityEstimator,
        base_loss: Optional[nn.Module] = None,
        adequacy_weight: float = 0.1,
        fluency_weight: float = 0.1,
        use_quality_weighting: bool = True,
    ):
        """
        Initialize combined quality loss.

        Args:
            quality_estimator: Quality estimator
            base_loss: Base MT loss (CrossEntropyLoss if None)
            adequacy_weight: Weight for adequacy loss
            fluency_weight: Weight for fluency loss
            use_quality_weighting: Whether to weight base loss by quality
        """
        super().__init__()
        self.quality_estimator = quality_estimator

        if use_quality_weighting:
            self.base_loss = QualityWeightedLoss(
                quality_estimator,
                base_loss
            )
        else:
            self.base_loss = base_loss or nn.CrossEntropyLoss()

        self.adequacy_loss = None
        if quality_estimator.config.use_adequacy:
            self.adequacy_loss = AdequacyLoss(quality_estimator)

        self.fluency_loss = None
        if quality_estimator.config.use_fluency:
            self.fluency_loss = FluencyLoss(quality_estimator)

        self.adequacy_weight = adequacy_weight
        self.fluency_weight = fluency_weight
        self.use_quality_weighting = use_quality_weighting

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        source_texts: List[str],
        translation_texts: List[str],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            logits: Model logits
            labels: Target labels
            source_texts: Source sentences
            translation_texts: Translation sentences

        Returns:
            Dictionary with loss components
        """
        losses = {}

        # Base MT loss
        if self.use_quality_weighting:
            base = self.base_loss(
                logits, labels, source_texts, translation_texts
            )
        else:
            base = self.base_loss(logits, labels)
        losses['base'] = base

        # Adequacy loss
        if self.adequacy_loss is not None:
            adequacy = self.adequacy_loss(source_texts, translation_texts)
            losses['adequacy'] = adequacy
        else:
            adequacy = 0.0

        # Fluency loss
        if self.fluency_loss is not None:
            fluency = self.fluency_loss(translation_texts)
            losses['fluency'] = fluency
        else:
            fluency = 0.0

        # Combined loss
        total = (
            base +
            self.adequacy_weight * adequacy +
            self.fluency_weight * fluency
        )
        losses['total'] = total

        return losses
