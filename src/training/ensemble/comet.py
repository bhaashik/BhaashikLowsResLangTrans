"""
COMET-based ensemble selection.

Uses the COMET (Crosslingual Optimized Metric for Evaluation of Translation)
neural quality estimation model to select the best translation from multiple
candidates.

COMET models are trained to predict human judgments of translation quality
and have shown strong correlation with human evaluation.

Reference:
    Rei et al. (2020). COMET: A Neural Framework for MT Evaluation.
    https://arxiv.org/abs/2009.09025
"""

from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass

from src.training.ensemble.base import (
    AbstractEnsemble,
    EnsembleResult,
    EnsembleConfig,
    select_by_scores,
)

logger = logging.getLogger(__name__)


class COMETEnsemble(AbstractEnsemble):
    """Ensemble method using COMET for candidate selection.

    COMET is a neural quality estimation model that predicts translation
    quality based on source, translation, and optionally reference.

    This ensemble method:
    1. Generates multiple candidate translations from each model
    2. Scores all candidates using COMET
    3. Selects the highest-scoring candidate for each source

    Expected improvement: +0.5 to +1.2 BLEU over single best model
    """

    def __init__(
        self,
        models: List[Any],
        config: Optional[EnsembleConfig] = None,
        comet_model_path: Optional[str] = None,
    ):
        """Initialize COMET ensemble.

        Args:
            models: List of translation models
            config: Ensemble configuration
            comet_model_path: Path to COMET model (overrides config)
        """
        super().__init__(models, config)

        self.comet_model_path = comet_model_path or self.config.comet_model
        self.comet_model = None
        self._load_comet_model()

    def _load_comet_model(self):
        """Load COMET model for quality estimation."""
        try:
            from comet import download_model, load_from_checkpoint

            logger.info(f"Loading COMET model: {self.comet_model_path}")

            # Download model if it's a known name
            if not self.comet_model_path.endswith(".ckpt"):
                model_path = download_model(self.comet_model_path)
            else:
                model_path = self.comet_model_path

            self.comet_model = load_from_checkpoint(model_path)
            self.comet_model.eval()

            # Move to specified device
            if self.config.device == "cuda":
                self.comet_model.cuda()

            logger.info("COMET model loaded successfully")

        except ImportError:
            logger.error(
                "COMET library not installed. Install with: pip install unbabel-comet"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load COMET model: {e}")
            raise

    def translate(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
        references: Optional[List[str]] = None,
        **kwargs
    ) -> EnsembleResult:
        """Translate texts using COMET-based selection.

        Args:
            texts: Source texts to translate
            source_lang: Source language code
            target_lang: Target language code
            references: Optional reference translations for scoring
            **kwargs: Additional arguments passed to models

        Returns:
            EnsembleResult with selected translations and scores
        """
        logger.info(
            f"Translating {len(texts)} texts with COMET ensemble "
            f"({len(self.models)} models, {self.config.num_candidates} candidates each)"
        )

        # Generate candidates from all models
        all_candidates = self.generate_candidates(
            texts, source_lang, target_lang
        )

        # Score candidates using COMET
        all_scores = self._score_candidates_comet(
            texts, all_candidates, references
        )

        # Select best candidate for each source
        translations = []
        selected_indices = []

        for candidates, scores in zip(all_candidates, all_scores):
            selected, idx, score = select_by_scores(
                candidates, scores, strategy="max"
            )
            translations.append(selected)
            selected_indices.append(idx)

        # Create result
        result = EnsembleResult(
            translations=translations,
            candidates=all_candidates,
            scores=all_scores,
            selected_indices=selected_indices,
            method="comet",
            metadata={
                "comet_model": self.comet_model_path,
                "num_models": len(self.models),
                "num_candidates": self.config.num_candidates,
                "temperature": self.config.temperature,
            }
        )

        logger.info(f"COMET ensemble translation completed")
        return result

    def _score_candidates_comet(
        self,
        sources: List[str],
        candidates_per_source: List[List[str]],
        references: Optional[List[str]] = None,
    ) -> List[List[float]]:
        """Score candidates using COMET.

        Args:
            sources: Source texts
            candidates_per_source: Candidates for each source
            references: Optional reference translations

        Returns:
            Scores for each candidate
        """
        all_scores = []

        # Prepare data for COMET
        comet_input = []
        for src_idx, (source, candidates) in enumerate(
            zip(sources, candidates_per_source)
        ):
            for candidate in candidates:
                # Check cache
                cached = self._get_cached_score(source, candidate, "comet")
                if cached is not None:
                    continue

                sample = {
                    "src": source,
                    "mt": candidate,
                }
                if references:
                    sample["ref"] = references[src_idx]

                comet_input.append(sample)

        # Score with COMET in batches
        if comet_input:
            logger.debug(f"Scoring {len(comet_input)} candidates with COMET")
            comet_scores = self._score_batch(comet_input)
        else:
            comet_scores = []

        # Organize scores per source
        score_idx = 0
        for src_idx, (source, candidates) in enumerate(
            zip(sources, candidates_per_source)
        ):
            scores = []
            for candidate in candidates:
                # Try cache first
                cached = self._get_cached_score(source, candidate, "comet")
                if cached is not None:
                    scores.append(cached)
                else:
                    score = comet_scores[score_idx]
                    scores.append(score)
                    self._cache_score(source, candidate, "comet", score)
                    score_idx += 1

            all_scores.append(scores)

        return all_scores

    def _score_batch(self, comet_input: List[Dict[str, str]]) -> List[float]:
        """Score a batch of candidates with COMET.

        Args:
            comet_input: List of dicts with 'src', 'mt', optionally 'ref'

        Returns:
            List of COMET scores
        """
        # Process in batches for efficiency
        batch_size = self.config.batch_size
        all_scores = []

        for i in range(0, len(comet_input), batch_size):
            batch = comet_input[i:i + batch_size]

            # COMET expects specific format
            model_output = self.comet_model.predict(
                batch,
                batch_size=len(batch),
                gpus=1 if self.config.device == "cuda" else 0,
            )

            # Extract scores (they're in 'scores' key)
            batch_scores = model_output.scores
            all_scores.extend(batch_scores)

        return all_scores

    def get_quality_estimates(
        self,
        sources: List[str],
        translations: List[str],
        references: Optional[List[str]] = None,
    ) -> List[float]:
        """Get COMET quality estimates for given translations.

        Useful for standalone quality estimation without ensemble.

        Args:
            sources: Source texts
            translations: Translations to score
            references: Optional reference translations

        Returns:
            COMET scores for each translation
        """
        comet_input = []
        for src_idx, (source, translation) in enumerate(zip(sources, translations)):
            sample = {
                "src": source,
                "mt": translation,
            }
            if references:
                sample["ref"] = references[src_idx]
            comet_input.append(sample)

        return self._score_batch(comet_input)
