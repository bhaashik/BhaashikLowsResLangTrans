#!/usr/bin/env python3
"""
CLI for merging multiple fine-tuned models.

Supports multiple merging strategies:
- TIES: Resolve conflicts via TRIM/ELECT/MERGE (expected +1.7 BLEU)
- DARE: Drop And REscale parameters (expected +1.9 BLEU)
- Model Soups: Uniform or greedy checkpoint averaging (expected +0.6-0.8 BLEU)

Usage:
    # Merge with TIES
    python scripts/merge_models.py \\
        --method ties \\
        --base-model facebook/nllb-200-distilled-600M \\
        --models models/nllb-bho models/nllb-mag models/nllb-awa \\
        --output models/nllb-merged-ties

    # Merge with DARE
    python scripts/merge_models.py \\
        --method dare \\
        --base-model facebook/nllb-200-distilled-600M \\
        --models models/nllb-bho models/nllb-mag \\
        --drop-rate 0.95 \\
        --output models/nllb-merged-dare

    # Create Model Soup from checkpoints
    python scripts/merge_models.py \\
        --method model_soups \\
        --models models/nllb-bho/checkpoint-*/ \\
        --soup-type greedy \\
        --eval-data data/val.tsv \\
        --output models/nllb-soup
"""

import argparse
import sys
from pathlib import Path
import logging
from typing import List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.merging import TIESMerger, DAREMerger, ModelSoups
from src.training.merging.base import MergeConfig
from src.training.models import ModelFactory, ModelConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Merge multiple fine-tuned models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Merging method
    parser.add_argument(
        '--method', required=True,
        choices=['ties', 'dare', 'model_soups'],
        help='Merging method'
    )

    # Model configuration
    parser.add_argument(
        '--base-model', required=True,
        help='Base model path or HuggingFace model name'
    )
    parser.add_argument(
        '--models', nargs='+', required=True,
        help='Paths to fine-tuned models or adapters to merge'
    )
    parser.add_argument(
        '--model-type',
        choices=['seq2seq', 'causal_lm'],
        default='seq2seq',
        help='Model type (default: seq2seq)'
    )

    # Output
    parser.add_argument(
        '--output', required=True,
        help='Output directory for merged model'
    )

    # TIES-specific
    ties_group = parser.add_argument_group('TIES Configuration')
    ties_group.add_argument(
        '--density', type=float, default=0.2,
        help='Density for TIES trimming (default: 0.2, keep top 20%%)'
    )

    # DARE-specific
    dare_group = parser.add_argument_group('DARE Configuration')
    dare_group.add_argument(
        '--drop-rate', type=float, default=0.95,
        help='Drop rate for DARE (default: 0.95, drop 95%%)'
    )

    # Model Soups-specific
    soup_group = parser.add_argument_group('Model Soups Configuration')
    soup_group.add_argument(
        '--soup-type',
        choices=['uniform', 'greedy'],
        default='uniform',
        help='Soup type (default: uniform)'
    )
    soup_group.add_argument(
        '--eval-data',
        help='Evaluation data for greedy soup (required for greedy soup)'
    )

    # General
    parser.add_argument(
        '--device',
        choices=['cuda', 'cpu'],
        default='cuda',
        help='Device (default: cuda)'
    )
    parser.add_argument(
        '--adapter-only', action='store_true',
        help='Models are LoRA adapters (not full models)'
    )

    return parser.parse_args()


def merge_with_ties(args):
    """Merge models using TIES method."""
    logger.info("Merging with TIES (Task Inference and Edit Selection)")
    logger.info(f"Density: {args.density}")

    config = MergeConfig(
        method="ties",
        density=args.density,
        device=args.device,
    )

    merger = TIESMerger(
        base_model_path=args.base_model,
        config=config
    )

    result = merger.merge(
        model_paths=args.models,
        output_path=args.output,
        adapter_only=args.adapter_only
    )

    logger.info(f"Merge completed: {result}")
    return result


def merge_with_dare(args):
    """Merge models using DARE method."""
    logger.info("Merging with DARE (Drop And REscale)")
    logger.info(f"Drop rate: {args.drop_rate}")

    config = MergeConfig(
        method="dare",
        drop_rate=args.drop_rate,
        device=args.device,
    )

    merger = DAREMerger(
        base_model_path=args.base_model,
        config=config
    )

    result = merger.merge(
        model_paths=args.models,
        output_path=args.output,
        adapter_only=args.adapter_only
    )

    logger.info(f"Merge completed: {result}")
    return result


def merge_with_model_soups(args):
    """Merge models using Model Soups."""
    logger.info(f"Creating Model Soup: {args.soup_type}")

    config = MergeConfig(
        method="model_soups",
        soup_type=args.soup_type,
        device=args.device,
    )

    merger = ModelSoups(config=config)

    # For greedy soup, need evaluation data and metric
    if args.soup_type == 'greedy':
        if not args.eval_data:
            raise ValueError("--eval-data required for greedy soup")

        # Load evaluation data
        from src.training.data import ParallelCorpusLoader
        loader = ParallelCorpusLoader(
            train_file=args.eval_data,
            format='tsv'
        )
        eval_dataset = loader.load()['train']

        # Define evaluation metric
        from src.training.trainers.metrics import compute_bleu
        def eval_metric_fn(state_dict, dataset):
            # This is simplified - would need to load model, translate, compute BLEU
            # For now, just return a placeholder
            logger.warning("Using placeholder evaluation - implement full metric")
            return 0.5

        result = merger.merge(
            model_paths=args.models,
            output_path=args.output,
            eval_dataset=eval_dataset,
            eval_metric_fn=eval_metric_fn
        )
    else:
        # Uniform soup
        result = merger.merge(
            model_paths=args.models,
            output_path=args.output,
        )

    logger.info(f"Merge completed: {result}")
    return result


def main():
    """Main merging function."""
    args = parse_args()

    logger.info("=" * 80)
    logger.info("Model Merging")
    logger.info("=" * 80)
    logger.info(f"Method: {args.method}")
    logger.info(f"Base model: {args.base_model}")
    logger.info(f"Models to merge: {len(args.models)}")
    for i, model_path in enumerate(args.models, 1):
        logger.info(f"  {i}. {model_path}")
    logger.info(f"Output: {args.output}")

    # Merge based on method
    if args.method == 'ties':
        result = merge_with_ties(args)
    elif args.method == 'dare':
        result = merge_with_dare(args)
    elif args.method == 'model_soups':
        result = merge_with_model_soups(args)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    logger.info("\n" + "=" * 80)
    logger.info("Merging completed successfully!")
    logger.info("=" * 80)
    logger.info(f"Merged model saved to: {args.output}")


if __name__ == '__main__':
    main()
