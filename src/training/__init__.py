"""
Training infrastructure for fine-tuning machine translation models.

This module provides:
- Model training with LoRA and full fine-tuning
- Model merging (TIES, DARE, Model Soups, etc.)
- Ensemble methods (COMET, MBR, Voting)
- Data loading and preprocessing
- Configurable training pipelines
"""

from .trainers import *
from .models import *
from .config import *

__version__ = "1.0.0"
