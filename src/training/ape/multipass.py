"""
Multi-pass APE.

Applies multiple refinement passes with different strategies
(beam search, sampling, etc.) and selects the best result.
"""

from typing import List, Optional, Dict, Tuple
import torch
import logging

from src.training.ape.config import APEConfig
from src.training.quality.quality_estimator import QualityEstimator

logger = logging.getLogger(__name__)


class MultiPassAPE:
    """
    Multi-pass automatic post-editing.

    Generates multiple refined versions using different strategies
    and selects the best one based on quality estimation.
    """

    def __init__(
        self,
        model,
        tokenizer,
        config: APEConfig,
        quality_estimator: Optional[QualityEstimator] = None,
    ):
        """
        Initialize multi-pass APE.

        Args:
            model: Translation model
            tokenizer: Tokenizer
            config: APE configuration
            quality_estimator: Quality estimator (required for selection)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.quality_estimator = quality_estimator

        if not quality_estimator:
            logger.warning(
                "MultiPassAPE works best with quality estimator for selection"
            )

    def refine_single(
        self,
        source_text: str,
        initial_translation: str,
        return_candidates: bool = False,
    ) -> str:
        """
        Refine translation using multiple passes.

        Args:
            source_text: Source sentence
            initial_translation: Initial translation
            return_candidates: If True, return (best, candidates)

        Returns:
            Best refined translation or (best, candidates) tuple
        """
        candidates = [
            {
                "pass": 0,
                "strategy": "initial",
                "translation": initial_translation,
                "quality": None,
            }
        ]

        # Estimate initial quality
        if self.quality_estimator:
            initial_quality = self.quality_estimator.estimate_single(
                source_text, initial_translation
            )
            candidates[0]["quality"] = initial_quality

            # Check if refinement needed
            if initial_quality >= self.config.quality_threshold:
                logger.debug(
                    f"Initial quality ({initial_quality:.3f}) above threshold, "
                    f"skipping APE"
                )
                if return_candidates:
                    return initial_translation, candidates
                else:
                    return initial_translation

        # Generate candidates with different strategies
        for pass_idx, strategy in enumerate(self.config.pass_strategies, 1):
            refined = self._generate_with_strategy(
                source_text,
                initial_translation,
                strategy,
                pass_idx
            )

            # Estimate quality
            quality = None
            if self.quality_estimator:
                quality = self.quality_estimator.estimate_single(
                    source_text, refined
                )

            candidates.append({
                "pass": pass_idx,
                "strategy": strategy,
                "translation": refined,
                "quality": quality,
            })

            logger.debug(
                f"Pass {pass_idx} ({strategy}): quality = {quality:.3f if quality else 'N/A'}"
            )

        # Select best candidate
        best_candidate = self._select_best_candidate(candidates)
        best_translation = best_candidate["translation"]

        logger.debug(
            f"Selected candidate from pass {best_candidate['pass']} "
            f"({best_candidate['strategy']})"
        )

        if return_candidates:
            return best_translation, candidates
        else:
            return best_translation

    def refine_batch(
        self,
        source_texts: List[str],
        initial_translations: List[str],
    ) -> List[str]:
        """
        Refine batch of translations.

        Args:
            source_texts: Source sentences
            initial_translations: Initial translations

        Returns:
            Refined translations
        """
        refined_translations = []

        for src, init_trans in zip(source_texts, initial_translations):
            refined = self.refine_single(src, init_trans)
            refined_translations.append(refined)

        return refined_translations

    def _generate_with_strategy(
        self,
        source: str,
        current_translation: str,
        strategy: str,
        pass_idx: int,
    ) -> str:
        """
        Generate refinement with specific strategy.

        Args:
            source: Source sentence
            current_translation: Current translation
            strategy: Generation strategy
            pass_idx: Pass index

        Returns:
            Refined translation
        """
        # Create prompt
        prompt = (
            f"Refine this translation (pass {pass_idx}):\n"
            f"Source: {source}\n"
            f"Current: {current_translation}\n"
            f"Refined:"
        )

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.config.device)

        # Generate with strategy-specific parameters
        gen_kwargs = {
            "max_length": 256,
            "early_stopping": True,
        }

        if strategy == "beam":
            gen_kwargs.update({
                "num_beams": self.config.beam_size,
                "length_penalty": self.config.length_penalty,
            })
        elif strategy == "sampling":
            gen_kwargs.update({
                "do_sample": True,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
            })
        elif strategy == "greedy":
            gen_kwargs.update({
                "num_beams": 1,
            })
        elif strategy == "diverse_beam":
            gen_kwargs.update({
                "num_beams": self.config.beam_size,
                "num_beam_groups": 2,
                "diversity_penalty": 1.0,
            })
        else:
            logger.warning(f"Unknown strategy: {strategy}, using beam search")
            gen_kwargs.update({
                "num_beams": self.config.beam_size,
            })

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        # Decode
        refined = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        # Extract refinement
        if "Refined:" in refined:
            refined = refined.split("Refined:")[-1].strip()

        return refined

    def _select_best_candidate(
        self,
        candidates: List[Dict],
    ) -> Dict:
        """
        Select best candidate based on quality.

        Args:
            candidates: List of candidate dictionaries

        Returns:
            Best candidate dictionary
        """
        if self.quality_estimator:
            # Select by quality
            candidates_with_quality = [
                c for c in candidates if c["quality"] is not None
            ]

            if candidates_with_quality:
                best = max(candidates_with_quality, key=lambda c: c["quality"])
                return best

        # No quality estimator or no quality scores, use last candidate
        logger.warning("No quality scores available, using last candidate")
        return candidates[-1]

    def get_statistics(
        self,
        source_texts: List[str],
        initial_translations: List[str],
    ) -> Dict[str, float]:
        """
        Get multi-pass APE statistics.

        Args:
            source_texts: Source sentences
            initial_translations: Initial translations

        Returns:
            Statistics dictionary
        """
        strategy_selections = {s: 0 for s in ["initial"] + self.config.pass_strategies}
        quality_improvements = []

        for src, init_trans in zip(source_texts, initial_translations):
            _, candidates = self.refine_single(
                src, init_trans, return_candidates=True
            )

            # Track which strategy was selected
            best = self._select_best_candidate(candidates)
            strategy_selections[best["strategy"]] += 1

            # Track quality improvement
            if candidates[0]["quality"] is not None and best["quality"] is not None:
                improvement = best["quality"] - candidates[0]["quality"]
                quality_improvements.append(improvement)

        stats = {
            "num_examples": len(source_texts),
            "num_passes": self.config.num_passes,
        }

        # Add strategy selection rates
        for strategy, count in strategy_selections.items():
            stats[f"selected_{strategy}"] = count
            stats[f"selected_{strategy}_rate"] = count / len(source_texts) if source_texts else 0.0

        if quality_improvements:
            stats.update({
                "avg_quality_improvement": sum(quality_improvements) / len(quality_improvements),
                "max_quality_improvement": max(quality_improvements),
                "min_quality_improvement": min(quality_improvements),
            })

        return stats
