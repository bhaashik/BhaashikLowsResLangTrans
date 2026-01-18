"""
Iterative refinement APE.

Uses the MT model itself to iteratively refine translations based on
quality feedback. Each iteration uses the previous translation as context.
"""

from typing import List, Optional, Dict, Tuple
import torch
import logging

from src.training.ape.config import APEConfig
from src.training.quality.quality_estimator import QualityEstimator

logger = logging.getLogger(__name__)


class IterativeRefinementAPE:
    """
    Iterative refinement automatic post-editing.

    Refines translations by iteratively re-translating with context
    from previous attempts. Uses quality estimation to guide refinement.
    """

    def __init__(
        self,
        model,
        tokenizer,
        config: APEConfig,
        quality_estimator: Optional[QualityEstimator] = None,
    ):
        """
        Initialize iterative refinement APE.

        Args:
            model: Translation model
            tokenizer: Tokenizer for the model
            config: APE configuration
            quality_estimator: Quality estimator (for quality-aware refinement)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.quality_estimator = quality_estimator

    def refine_single(
        self,
        source_text: str,
        initial_translation: str,
        return_history: bool = False,
    ) -> str:
        """
        Refine single translation.

        Args:
            source_text: Source sentence
            initial_translation: Initial translation to refine
            return_history: If True, return (final, history)

        Returns:
            Refined translation or (final, history) tuple
        """
        current_translation = initial_translation
        history = [{"iteration": 0, "translation": initial_translation, "quality": None}]

        # Get initial quality
        if self.quality_estimator:
            initial_quality = self.quality_estimator.estimate_single(
                source_text, initial_translation
            )
            history[0]["quality"] = initial_quality

            # Check if refinement needed
            if initial_quality >= self.config.quality_threshold:
                logger.debug(
                    f"Translation quality ({initial_quality:.3f}) "
                    f"above threshold ({self.config.quality_threshold:.3f}), "
                    f"skipping APE"
                )
                if return_history:
                    return current_translation, history
                else:
                    return current_translation

        # Iterative refinement
        for iteration in range(1, self.config.max_iterations + 1):
            # Create refinement prompt
            refinement_prompt = self._create_refinement_prompt(
                source_text,
                current_translation,
                iteration
            )

            # Generate refined translation
            refined = self._generate_refinement(refinement_prompt)

            # Estimate quality
            if self.quality_estimator:
                refined_quality = self.quality_estimator.estimate_single(
                    source_text, refined
                )

                # Check improvement
                previous_quality = history[-1]["quality"]
                improvement = refined_quality - previous_quality

                logger.debug(
                    f"Iteration {iteration}: "
                    f"quality {previous_quality:.3f} → {refined_quality:.3f} "
                    f"(Δ {improvement:+.3f})"
                )

                # Record history
                history.append({
                    "iteration": iteration,
                    "translation": refined,
                    "quality": refined_quality,
                    "improvement": improvement,
                })

                # Check stopping criteria
                if improvement < self.config.min_quality_improvement:
                    logger.debug(
                        f"Improvement ({improvement:.3f}) below threshold "
                        f"({self.config.min_quality_improvement:.3f}), stopping"
                    )
                    break

                if refined_quality >= self.config.quality_threshold:
                    logger.debug(
                        f"Quality ({refined_quality:.3f}) above threshold "
                        f"({self.config.quality_threshold:.3f}), stopping"
                    )
                    current_translation = refined
                    break

                # Update current
                if refined_quality > previous_quality:
                    current_translation = refined
                else:
                    logger.debug("No quality improvement, stopping")
                    break
            else:
                # No quality estimation, just use refined
                history.append({
                    "iteration": iteration,
                    "translation": refined,
                    "quality": None,
                })
                current_translation = refined

        if return_history:
            return current_translation, history
        else:
            return current_translation

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

    def _create_refinement_prompt(
        self,
        source: str,
        current_translation: str,
        iteration: int,
    ) -> str:
        """
        Create refinement prompt for model.

        Args:
            source: Source sentence
            current_translation: Current translation
            iteration: Current iteration number

        Returns:
            Refinement prompt
        """
        # Simple approach: concatenate source with current translation
        # More sophisticated: add special tokens or instructions
        if iteration == 1:
            # First refinement: simple prompt
            prompt = f"Improve this translation:\nSource: {source}\nTranslation: {current_translation}\nImproved:"
        else:
            # Later refinements: emphasize improvement
            prompt = f"Further improve this translation:\nSource: {source}\nCurrent: {current_translation}\nBetter:"

        return prompt

    def _generate_refinement(self, prompt: str) -> str:
        """
        Generate refined translation from prompt.

        Args:
            prompt: Refinement prompt

        Returns:
            Refined translation
        """
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.config.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=256,
                num_beams=self.config.beam_size,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                length_penalty=self.config.length_penalty,
                early_stopping=True,
            )

        # Decode
        refined = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        # Extract just the translation (remove prompt)
        # This is a simple heuristic - may need adjustment
        if "Improved:" in refined:
            refined = refined.split("Improved:")[-1].strip()
        elif "Better:" in refined:
            refined = refined.split("Better:")[-1].strip()

        return refined

    def get_statistics(
        self,
        source_texts: List[str],
        initial_translations: List[str],
    ) -> Dict[str, float]:
        """
        Get APE statistics for a dataset.

        Args:
            source_texts: Source sentences
            initial_translations: Initial translations

        Returns:
            Statistics dictionary
        """
        num_refined = 0
        total_iterations = 0
        quality_improvements = []

        for src, init_trans in zip(source_texts, initial_translations):
            if self.quality_estimator:
                initial_quality = self.quality_estimator.estimate_single(src, init_trans)

                if initial_quality < self.config.quality_threshold:
                    refined, history = self.refine_single(
                        src, init_trans, return_history=True
                    )
                    num_refined += 1
                    total_iterations += len(history) - 1

                    final_quality = history[-1]["quality"]
                    improvement = final_quality - initial_quality
                    quality_improvements.append(improvement)

        stats = {
            "num_examples": len(source_texts),
            "num_refined": num_refined,
            "refinement_rate": num_refined / len(source_texts) if source_texts else 0.0,
            "avg_iterations": total_iterations / num_refined if num_refined > 0 else 0.0,
        }

        if quality_improvements:
            stats.update({
                "avg_quality_improvement": sum(quality_improvements) / len(quality_improvements),
                "max_quality_improvement": max(quality_improvements),
                "min_quality_improvement": min(quality_improvements),
            })

        return stats
