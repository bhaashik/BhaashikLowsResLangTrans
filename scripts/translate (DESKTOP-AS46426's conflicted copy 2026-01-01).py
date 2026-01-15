#!/usr/bin/env python3
"""
Main translation script using tiered orchestrator.

Usage:
    # Translate using tiered strategy (70/20/10)
    python scripts/translate.py --src en --tgt bho --input texts.txt --tiered

    # Translate using only IndicTrans2
    python scripts/translate.py --src en --tgt hi --input texts.txt --indictrans2

    # Get cost estimate
    python scripts/translate.py --src en --tgt bho --num-samples 10000 --estimate-only
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tiered_orchestrator import TieredOrchestrator
from src.translators import IndicTrans2Translator, NLLBTranslator, HindiPivotTranslator
from src.utils.config import Config
from src.utils.logger import setup_logger, get_logger


logger = get_logger(__name__)


def load_texts(input_file: str) -> list:
    """Load texts from file."""
    with open(input_file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    return texts


def save_translations(translations: list, output_file: str):
    """Save translations to file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for trans in translations:
            f.write(trans + '\n')


def main():
    parser = argparse.ArgumentParser(
        description="Translate texts using BhaashikLowsResLangTrans"
    )

    # Language options
    parser.add_argument(
        "--src",
        required=True,
        help="Source language code (e.g., en, hi)"
    )
    parser.add_argument(
        "--tgt",
        required=True,
        help="Target language code (e.g., bho, hi)"
    )

    # Input/Output
    parser.add_argument(
        "--input",
        type=str,
        help="Input file with source texts (one per line)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for translations"
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Single text to translate (alternative to --input)"
    )

    # Translation method
    parser.add_argument(
        "--tiered",
        action="store_true",
        help="Use tiered translation (70/20/10 strategy)"
    )
    parser.add_argument(
        "--indictrans2",
        action="store_true",
        help="Use IndicTrans2 only"
    )
    parser.add_argument(
        "--nllb",
        action="store_true",
        help="Use NLLB only"
    )
    parser.add_argument(
        "--hindi-pivot",
        action="store_true",
        help="Use Hindi pivot translation"
    )

    # Tiered strategy options
    parser.add_argument(
        "--tier-free",
        type=float,
        default=70,
        help="Percentage for free tier (default: 70)"
    )
    parser.add_argument(
        "--tier-enhancement",
        type=float,
        default=20,
        help="Percentage for enhancement tier (default: 20)"
    )
    parser.add_argument(
        "--tier-premium",
        type=float,
        default=10,
        help="Percentage for premium tier (default: 10)"
    )

    # Cost estimation
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Only estimate cost, don't translate"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        help="Number of samples for cost estimation"
    )

    # Other options
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for tier assignment"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config file"
    )
    parser.add_argument(
        "--log-level",
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help="Logging level"
    )

    args = parser.parse_args()

    # Initialize configuration
    config = Config(args.config) if args.config else Config()

    # Setup logging
    setup_logger(
        log_file=config.log_file,
        level=args.log_level,
        console=True
    )

    logger.info("="*70)
    logger.info("BHAASHIK LOW-RESOURCE LANGUAGE TRANSLATION")
    logger.info("="*70)
    logger.info(f"Source: {args.src} → Target: {args.tgt}")
    logger.info("="*70)

    # Cost estimation only
    if args.estimate_only:
        if not args.num_samples:
            logger.error("--num-samples required for cost estimation")
            return

        orchestrator = TieredOrchestrator(
            config=config,
            tier_percentages={
                'free': args.tier_free,
                'enhancement': args.tier_enhancement,
                'premium': args.tier_premium
            }
        )

        orchestrator.print_cost_estimate(args.num_samples)
        return

    # Load texts
    if args.text:
        texts = [args.text]
    elif args.input:
        texts = load_texts(args.input)
        logger.info(f"Loaded {len(texts):,} texts from {args.input}")
    else:
        logger.error("Either --text or --input required")
        parser.print_help()
        return

    # Translate
    translations = None

    if args.tiered:
        # Use tiered orchestrator
        orchestrator = TieredOrchestrator(
            config=config,
            tier_percentages={
                'free': args.tier_free,
                'enhancement': args.tier_enhancement,
                'premium': args.tier_premium
            }
        )

        results = orchestrator.translate(
            texts,
            src_lang=args.src,
            tgt_lang=args.tgt,
            batch_size=args.batch_size,
            seed=args.seed
        )

        translations = results['translations']

    elif args.indictrans2:
        # Use IndicTrans2 only
        translator = IndicTrans2Translator(config=config)
        translations = translator.translate(
            texts,
            src_lang=args.src,
            tgt_lang=args.tgt,
            batch_size=args.batch_size
        )

    elif args.nllb:
        # Use NLLB only
        translator = NLLBTranslator(config=config)
        translations = translator.translate(
            texts,
            src_lang=args.src,
            tgt_lang=args.tgt,
            batch_size=args.batch_size
        )

    elif args.hindi_pivot:
        # Use Hindi pivot
        translator = HindiPivotTranslator(config=config)
        translations = translator.translate(
            texts,
            src_lang=args.src,
            tgt_lang=args.tgt,
            batch_size=args.batch_size
        )

    else:
        # Default: use tiered strategy
        logger.info("No method specified, using tiered strategy")
        orchestrator = TieredOrchestrator(config=config)
        results = orchestrator.translate(
            texts,
            src_lang=args.src,
            tgt_lang=args.tgt,
            batch_size=args.batch_size,
            seed=args.seed
        )
        translations = results['translations']

    # Save or print
    if args.output:
        save_translations(translations, args.output)
        logger.success(f"✓ Translations saved to {args.output}")
    elif args.text:
        print("\nTranslation:")
        print(translations[0])
    else:
        # Print first few
        print("\nSample translations:")
        for i, (src, tgt) in enumerate(zip(texts[:5], translations[:5])):
            print(f"\n[{i+1}]")
            print(f"  Source: {src}")
            print(f"  Translation: {tgt}")

    logger.info("\n" + "="*70)
    logger.info("TRANSLATION COMPLETE")
    logger.info("="*70)


if __name__ == "__main__":
    main()
