"""
Back-translation APE.

Uses round-trip translation (source → target → source) to verify quality
and identify issues. Refines based on back-translation quality.
"""

from typing import List, Optional, Dict
import torch
import logging

from src.training.ape.config import APEConfig
from src.training.quality.quality_estimator import QualityEstimator

logger = logging.getLogger(__name__)


class BackTranslationAPE:
    """
    Back-translation automatic post-editing.

    Verifies translation quality via round-trip translation and refines
    if back-translation doesn't match source.
    """

    def __init__(
        self,
        forward_model,
        forward_tokenizer,
        backward_model,
        backward_tokenizer,
        config: APEConfig,
        quality_estimator: Optional[QualityEstimator] = None,
    ):
        """
        Initialize back-translation APE.

        Args:
            forward_model: Source→Target translation model
            forward_tokenizer: Tokenizer for forward model
            backward_model: Target→Source back-translation model
            backward_tokenizer: Tokenizer for backward model
            config: APE configuration
            quality_estimator: Quality estimator
        """
        self.forward_model = forward_model
        self.forward_tokenizer = forward_tokenizer
        self.backward_model = backward_model
        self.backward_tokenizer = backward_tokenizer
        self.config = config
        self.quality_estimator = quality_estimator

    def refine_single(
        self,
        source_text: str,
        initial_translation: str,
        return_diagnostics: bool = False,
    ) -> str:
        """
        Refine translation using back-translation.

        Args:
            source_text: Source sentence
            initial_translation: Initial translation
            return_diagnostics: If True, return (refined, diagnostics)

        Returns:
            Refined translation or (refined, diagnostics) tuple
        """
        diagnostics = {
            "initial_translation": initial_translation,
            "back_translation": None,
            "round_trip_quality": None,
            "refined": None,
        }

        # Back-translate
        back_translation = self._back_translate(initial_translation)
        diagnostics["back_translation"] = back_translation

        # Compute round-trip quality (source vs back-translation)
        if self.quality_estimator:
            round_trip_quality = self.quality_estimator.estimate_single(
                source_text,
                back_translation
            )
            diagnostics["round_trip_quality"] = round_trip_quality

            logger.debug(
                f"Round-trip quality: {round_trip_quality:.3f} "
                f"(threshold: {self.config.quality_threshold:.3f})"
            )

            # If round-trip quality is good, keep initial translation
            if round_trip_quality >= self.config.quality_threshold:
                logger.debug("Round-trip quality good, keeping initial translation")
                diagnostics["refined"] = initial_translation

                if return_diagnostics:
                    return initial_translation, diagnostics
                else:
                    return initial_translation

        # Round-trip quality is poor, try to refine
        # Create a prompt that incorporates back-translation feedback
        refined = self._refine_with_backtranslation_feedback(
            source_text,
            initial_translation,
            back_translation
        )

        diagnostics["refined"] = refined

        if return_diagnostics:
            return refined, diagnostics
        else:
            return refined

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

    def _back_translate(self, translation: str) -> str:
        """
        Back-translate target to source.

        Args:
            translation: Target language text

        Returns:
            Back-translated source
        """
        # Tokenize
        inputs = self.backward_tokenizer(
            translation,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.config.device)

        # Generate
        with torch.no_grad():
            outputs = self.backward_model.generate(
                **inputs,
                max_length=256,
                num_beams=self.config.beam_size,
                length_penalty=self.config.length_penalty,
                early_stopping=True,
            )

        # Decode
        back_translation = self.backward_tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        return back_translation

    def _refine_with_backtranslation_feedback(
        self,
        source: str,
        initial_translation: str,
        back_translation: str,
    ) -> str:
        """
        Refine translation using back-translation feedback.

        Args:
            source: Original source
            initial_translation: Initial translation
            back_translation: Back-translated source

        Returns:
            Refined translation
        """
        # Create feedback prompt
        # Highlight the discrepancy between source and back-translation
        prompt = (
            f"The translation may have issues. "
            f"Original: {source}. "
            f"Back-translation: {back_translation}. "
            f"Improve this translation: {initial_translation}. "
            f"Corrected:"
        )

        # Generate refined translation
        inputs = self.forward_tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self.config.device)

        with torch.no_grad():
            outputs = self.forward_model.generate(
                **inputs,
                max_length=256,
                num_beams=self.config.beam_size,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                length_penalty=self.config.length_penalty,
                early_stopping=True,
            )

        refined = self.forward_tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        # Extract refined translation (remove prompt)
        if "Corrected:" in refined:
            refined = refined.split("Corrected:")[-1].strip()

        return refined

    def get_statistics(
        self,
        source_texts: List[str],
        initial_translations: List[str],
    ) -> Dict[str, float]:
        """
        Get back-translation APE statistics.

        Args:
            source_texts: Source sentences
            initial_translations: Initial translations

        Returns:
            Statistics dictionary
        """
        num_refined = 0
        round_trip_qualities = []

        for src, init_trans in zip(source_texts, initial_translations):
            _, diagnostics = self.refine_single(
                src, init_trans, return_diagnostics=True
            )

            if diagnostics["round_trip_quality"] is not None:
                round_trip_qualities.append(diagnostics["round_trip_quality"])

                if diagnostics["round_trip_quality"] < self.config.quality_threshold:
                    num_refined += 1

        stats = {
            "num_examples": len(source_texts),
            "num_refined": num_refined,
            "refinement_rate": num_refined / len(source_texts) if source_texts else 0.0,
        }

        if round_trip_qualities:
            stats.update({
                "avg_round_trip_quality": sum(round_trip_qualities) / len(round_trip_qualities),
                "min_round_trip_quality": min(round_trip_qualities),
                "max_round_trip_quality": max(round_trip_qualities),
            })

        return stats
