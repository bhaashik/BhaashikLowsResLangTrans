"""Automatic Post-Editing (APE) for machine translation.

This module provides automatic post-editing mechanisms to refine translations:
- Iterative refinement: Re-translate with context from previous attempts
- Back-translation: Verify quality via round-trip translation
- Multi-pass refinement: Multiple editing passes with different strategies
- Quality-aware APE: Only refine translations below quality threshold

APE is particularly useful for:
1. Improving low-quality translations
2. Fixing systematic errors
3. Adapting to domain-specific requirements
4. Bootstrapping training data

References:
- Junczys-Dowmunt & Grundkiewicz (2016): Log-linear Combinations of APE
- Chatterjee et al. (2020): Findings of WMT20 APE Shared Task
- Freitag et al. (2019): APE at Scale and its Implications
"""

from src.training.ape.config import APEConfig
from src.training.ape.iterative import IterativeRefinementAPE
from src.training.ape.backtranslation import BackTranslationAPE
from src.training.ape.multipass import MultiPassAPE
from src.training.ape.ape_pipeline import APEPipeline, create_ape_pipeline

__all__ = [
    "APEConfig",
    "IterativeRefinementAPE",
    "BackTranslationAPE",
    "MultiPassAPE",
    "APEPipeline",
    "create_ape_pipeline",
]
