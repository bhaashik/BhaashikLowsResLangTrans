"""
APE Pipeline.

Unified interface for automatic post-editing methods.
"""

from typing import List, Optional, Dict
import logging

from src.training.ape.config import APEConfig
from src.training.ape.iterative import IterativeRefinementAPE
from src.training.ape.backtranslation import BackTranslationAPE
from src.training.ape.multipass import MultiPassAPE
from src.training.quality.quality_estimator import QualityEstimator

logger = logging.getLogger(__name__)


class APEPipeline:
    """
    Automatic Post-Editing Pipeline.

    Provides unified interface for different APE methods.
    """

    def __init__(
        self,
        config: APEConfig,
        forward_model,
        forward_tokenizer,
        backward_model=None,
        backward_tokenizer=None,
        quality_estimator: Optional[QualityEstimator] = None,
    ):
        """
        Initialize APE pipeline.

        Args:
            config: APE configuration
            forward_model: Source→Target model
            forward_tokenizer: Tokenizer for forward model
            backward_model: Target→Source model (for back-translation)
            backward_tokenizer: Tokenizer for backward model
            quality_estimator: Quality estimator
        """
        self.config = config
        self.forward_model = forward_model
        self.forward_tokenizer = forward_tokenizer
        self.backward_model = backward_model
        self.backward_tokenizer = backward_tokenizer
        self.quality_estimator = quality_estimator

        self.ape_engine = None
        self._initialize_engine()

    def _initialize_engine(self):
        """Initialize APE engine based on method."""
        logger.info(f"Initializing APE engine: {self.config.ape_method}")

        if self.config.ape_method == "iterative":
            self.ape_engine = IterativeRefinementAPE(
                model=self.forward_model,
                tokenizer=self.forward_tokenizer,
                config=self.config,
                quality_estimator=self.quality_estimator,
            )

        elif self.config.ape_method == "backtranslation":
            if not self.backward_model or not self.backward_tokenizer:
                raise ValueError(
                    "Back-translation APE requires backward_model and backward_tokenizer"
                )
            self.ape_engine = BackTranslationAPE(
                forward_model=self.forward_model,
                forward_tokenizer=self.forward_tokenizer,
                backward_model=self.backward_model,
                backward_tokenizer=self.backward_tokenizer,
                config=self.config,
                quality_estimator=self.quality_estimator,
            )

        elif self.config.ape_method == "multipass":
            self.ape_engine = MultiPassAPE(
                model=self.forward_model,
                tokenizer=self.forward_tokenizer,
                config=self.config,
                quality_estimator=self.quality_estimator,
            )

        else:
            raise ValueError(f"Unknown APE method: {self.config.ape_method}")

        logger.info("✓ APE engine initialized")

    def refine(
        self,
        source_texts: List[str],
        translations: List[str],
    ) -> List[str]:
        """
        Refine translations using APE.

        Args:
            source_texts: Source sentences
            translations: Initial translations

        Returns:
            Refined translations
        """
        if not self.config.use_ape:
            logger.debug("APE disabled, returning original translations")
            return translations

        logger.info(f"Refining {len(translations)} translations with APE...")

        refined = self.ape_engine.refine_batch(source_texts, translations)

        logger.info("✓ APE refinement complete")
        return refined

    def refine_single(
        self,
        source_text: str,
        translation: str,
    ) -> str:
        """
        Refine single translation.

        Args:
            source_text: Source sentence
            translation: Initial translation

        Returns:
            Refined translation
        """
        if not self.config.use_ape:
            return translation

        refined = self.ape_engine.refine_single(source_text, translation)
        return refined

    def get_statistics(
        self,
        source_texts: List[str],
        translations: List[str],
    ) -> Dict[str, float]:
        """
        Get APE statistics.

        Args:
            source_texts: Source sentences
            translations: Initial translations

        Returns:
            Statistics dictionary
        """
        if not self.config.use_ape:
            return {"ape_enabled": False}

        stats = self.ape_engine.get_statistics(source_texts, translations)
        stats["ape_enabled"] = True
        stats["ape_method"] = self.config.ape_method
        return stats


def create_ape_pipeline(
    config: APEConfig,
    forward_model,
    forward_tokenizer,
    backward_model=None,
    backward_tokenizer=None,
    quality_estimator: Optional[QualityEstimator] = None,
) -> APEPipeline:
    """
    Create APE pipeline.

    Args:
        config: APE configuration
        forward_model: Source→Target model
        forward_tokenizer: Tokenizer for forward model
        backward_model: Target→Source model (optional)
        backward_tokenizer: Tokenizer for backward model (optional)
        quality_estimator: Quality estimator (optional but recommended)

    Returns:
        APEPipeline instance
    """
    pipeline = APEPipeline(
        config=config,
        forward_model=forward_model,
        forward_tokenizer=forward_tokenizer,
        backward_model=backward_model,
        backward_tokenizer=backward_tokenizer,
        quality_estimator=quality_estimator,
    )

    return pipeline
