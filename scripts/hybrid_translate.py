#!/usr/bin/env python3
"""
CLI for hybrid translation with quality-based routing.

Combines fine-tuned models, API translation, and ensemble methods
for optimal quality/cost tradeoffs.

Usage:
    # Basic hybrid translation
    python scripts/hybrid_translate.py \\
        --finetuned-model models/nllb-bhojpuri \\
        --source-lang hi \\
        --target-lang bho \\
        --input texts.txt \\
        --output translations.txt

    # With API fallback
    python scripts/hybrid_translate.py \\
        --finetuned-model models/nllb-bhojpuri \\
        --api-provider openai \\
        --api-model gpt-4o-mini \\
        --source-lang hi \\
        --target-lang bho \\
        --input texts.txt \\
        --output translations.txt \\
        --quality-threshold 0.7

    # With ensemble for critical translations
    python scripts/hybrid_translate.py \\
        --finetuned-model models/nllb-bho \\
        --ensemble-models models/nllb-bho models/mbart-bho models/mt5-bho \\
        --ensemble-method comet \\
        --source-lang hi \\
        --target-lang bho \\
        --input texts.txt \\
        --output translations.txt

    # With COMET quality estimation
    python scripts/hybrid_translate.py \\
        --finetuned-model models/nllb-bho \\
        --api-provider anthropic \\
        --use-comet-qe \\
        --source-lang hi \\
        --target-lang bho \\
        --text "नमस्ते, आप कैसे हैं?"
"""

import argparse
import sys
from pathlib import Path
import logging
from typing import List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.pipeline import HybridPipeline, HybridConfig
from universal_translate.providers import FineTunedProvider, ProviderRegistry

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Hybrid translation with quality-based routing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Model configuration
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument(
        '--finetuned-model', required=True,
        help='Path to fine-tuned model or adapter'
    )
    model_group.add_argument(
        '--base-model',
        help='Base model name (required if finetuned-model is adapter only)'
    )
    model_group.add_argument(
        '--adapter-only', action='store_true',
        help='Fine-tuned model is LoRA adapter only'
    )

    # Language configuration
    lang_group = parser.add_argument_group('Language Configuration')
    lang_group.add_argument(
        '--source-lang', required=True,
        help='Source language code (e.g., hi, en)'
    )
    lang_group.add_argument(
        '--target-lang', required=True,
        help='Target language code (e.g., bho, mwr)'
    )

    # Input/Output
    io_group = parser.add_argument_group('Input/Output')
    io_group.add_argument(
        '--text',
        help='Single text to translate'
    )
    io_group.add_argument(
        '--input',
        help='Input file (one text per line)'
    )
    io_group.add_argument(
        '--output',
        help='Output file (translations, one per line)'
    )

    # API fallback configuration
    api_group = parser.add_argument_group('API Fallback Configuration')
    api_group.add_argument(
        '--api-provider',
        choices=['openai', 'anthropic', 'gemini', 'vertex'],
        help='API provider for fallback'
    )
    api_group.add_argument(
        '--api-model',
        help='API model name'
    )
    api_group.add_argument(
        '--api-key',
        help='API key (or use environment variable)'
    )

    # Ensemble configuration
    ensemble_group = parser.add_argument_group('Ensemble Configuration')
    ensemble_group.add_argument(
        '--ensemble-models', nargs='+',
        help='Paths to models for ensemble'
    )
    ensemble_group.add_argument(
        '--ensemble-method',
        choices=['comet', 'mbr', 'voting'],
        default='comet',
        help='Ensemble method (default: comet)'
    )

    # Hybrid pipeline configuration
    hybrid_group = parser.add_argument_group('Hybrid Pipeline Configuration')
    hybrid_group.add_argument(
        '--finetuned-percentage', type=float, default=90.0,
        help='Target percentage for fine-tuned model (default: 90.0)'
    )
    hybrid_group.add_argument(
        '--api-percentage', type=float, default=9.0,
        help='Target percentage for API fallback (default: 9.0)'
    )
    hybrid_group.add_argument(
        '--ensemble-percentage', type=float, default=1.0,
        help='Target percentage for ensemble (default: 1.0)'
    )
    hybrid_group.add_argument(
        '--quality-threshold', type=float, default=0.7,
        help='Quality threshold for API fallback (default: 0.7)'
    )
    hybrid_group.add_argument(
        '--ensemble-threshold', type=float, default=0.5,
        help='Quality threshold for ensemble (default: 0.5)'
    )
    hybrid_group.add_argument(
        '--use-comet-qe', action='store_true',
        help='Use COMET-QE for quality estimation (slower but more accurate)'
    )
    hybrid_group.add_argument(
        '--force-percentages', action='store_true',
        help='Force exact percentages (vs quality-based routing)'
    )

    # Performance
    parser.add_argument(
        '--batch-size', type=int, default=16,
        help='Batch size for fine-tuned model (default: 16)'
    )
    parser.add_argument(
        '--device',
        choices=['cuda', 'cpu'],
        default='cuda',
        help='Device (default: cuda)'
    )

    return parser.parse_args()


def load_input_texts(args) -> List[str]:
    """Load input texts from command line or file."""
    if args.text:
        return [args.text]
    elif args.input:
        with open(args.input, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    else:
        raise ValueError("Either --text or --input required")


def save_translations(translations: List[str], output_path: str):
    """Save translations to file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for translation in translations:
            f.write(translation + '\n')


def create_finetuned_provider(args):
    """Create fine-tuned model provider."""
    logger.info(f"Loading fine-tuned model from {args.finetuned_model}")

    provider = FineTunedProvider(
        model_path=args.finetuned_model,
        base_model=args.base_model,
        adapter_only=args.adapter_only,
        device=args.device,
        batch_size=args.batch_size,
    )

    provider.initialize()
    return provider


def create_api_provider(args):
    """Create API provider for fallback."""
    if not args.api_provider:
        return None

    logger.info(f"Setting up API provider: {args.api_provider}")

    api_provider = ProviderRegistry.create_provider(
        provider=args.api_provider,
        model=args.api_model,
        api_key=args.api_key,
    )

    return api_provider


def create_ensemble_models(args):
    """Create ensemble models."""
    if not args.ensemble_models:
        return None

    logger.info(f"Loading {len(args.ensemble_models)} models for ensemble")

    ensemble_models = []
    for model_path in args.ensemble_models:
        provider = FineTunedProvider(
            model_path=model_path,
            base_model=args.base_model,
            adapter_only=args.adapter_only,
            device=args.device,
        )
        provider.initialize()
        # Extract the actual model
        ensemble_models.append(provider.model)

    return ensemble_models


def main():
    """Main hybrid translation function."""
    args = parse_args()

    logger.info("=" * 80)
    logger.info("Hybrid Translation Pipeline")
    logger.info("=" * 80)

    # Load input texts
    texts = load_input_texts(args)
    logger.info(f"Loaded {len(texts)} texts to translate")

    # Create providers
    finetuned_provider = create_finetuned_provider(args)
    api_provider = create_api_provider(args)
    ensemble_models = create_ensemble_models(args)

    # Create hybrid configuration
    config = HybridConfig(
        finetuned_percentage=args.finetuned_percentage,
        api_percentage=args.api_percentage,
        ensemble_percentage=args.ensemble_percentage,
        quality_threshold=args.quality_threshold,
        ensemble_threshold=args.ensemble_threshold,
        use_comet_qe=args.use_comet_qe,
        force_percentages=args.force_percentages,
        api_provider=args.api_provider or "none",
        api_model=args.api_model or "none",
        ensemble_method=args.ensemble_method,
    )

    logger.info(f"Hybrid config: {config.finetuned_percentage}% / "
                f"{config.api_percentage}% / {config.ensemble_percentage}%")

    # Create pipeline
    pipeline = HybridPipeline(
        finetuned_model=finetuned_provider,
        api_model=api_provider,
        ensemble_models=ensemble_models,
        config=config,
    )

    # Translate
    logger.info("\n" + "=" * 80)
    logger.info("Translating...")
    logger.info("=" * 80)

    result = pipeline.translate(
        texts=texts,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
    )

    # Print results
    logger.info("\n" + "=" * 80)
    logger.info("Translation completed!")
    logger.info("=" * 80)

    # Print routing statistics
    route_stats = result.get_route_statistics()
    logger.info("\nRouting statistics:")
    for route, stats in route_stats.items():
        logger.info(f"  {route}: {stats['count']} ({stats['percentage']:.1f}%)")

    # Print cost
    logger.info(f"\nTotal cost: ${result.costs['total']:.4f}")
    logger.info(f"Cost per translation: ${result.costs['per_translation']:.6f}")

    # Save or print translations
    if args.output:
        save_translations(result.translations, args.output)
        logger.info(f"\nTranslations saved to: {args.output}")
    else:
        logger.info("\nTranslations:")
        for i, (source, translation) in enumerate(zip(texts, result.translations), 1):
            logger.info(f"{i}. {source}")
            logger.info(f"   → {translation}")
            logger.info(f"   Route: {result.routes[i-1].value}, "
                       f"Quality: {result.quality_estimates[i-1].score:.3f}")

    # Print pipeline statistics
    logger.info("\nPipeline statistics:")
    stats = pipeline.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.2f}")
        else:
            logger.info(f"  {key}: {value}")


if __name__ == '__main__':
    main()
