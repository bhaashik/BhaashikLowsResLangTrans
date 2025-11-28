#!/usr/bin/env python3
"""
Main download script for datasets and models.

Usage:
    python scripts/download.py --samanantar --all-languages
    python scripts/download.py --models indictrans2 nllb
    python scripts/download.py --everything
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.downloaders import SamanantarDownloader, ModelDownloader
from src.utils.config import Config
from src.utils.logger import setup_logger, get_logger


logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Download datasets and models for BhaashikLowsResLangTrans"
    )

    # Dataset options
    parser.add_argument(
        "--samanantar",
        action="store_true",
        help="Download Samanantar corpus"
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        help="Specific languages to download (e.g., hi bn ta)"
    )
    parser.add_argument(
        "--all-languages",
        action="store_true",
        help="Download all available languages"
    )

    # Model options
    parser.add_argument(
        "--models",
        nargs="+",
        choices=['indictrans2', 'nllb', 'llama', 'mistral', 'aya'],
        help="Models to download"
    )

    # Convenience options
    parser.add_argument(
        "--essential",
        action="store_true",
        help="Download essential resources (Samanantar + IndicTrans2 + NLLB)"
    )
    parser.add_argument(
        "--everything",
        action="store_true",
        help="Download everything including LLMs (requires ~600GB)"
    )

    # Configuration
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
    config.ensure_directories()

    # Setup logging
    setup_logger(
        log_file=config.log_file,
        level=args.log_level,
        console=True
    )

    logger.info("="*70)
    logger.info("BHAASHIK LOW-RESOURCE LANGUAGE TRANSLATION")
    logger.info("DATA DOWNLOAD SCRIPT")
    logger.info("="*70)
    logger.info(f"Base directory: {config.base_dir}")
    logger.info("="*70)

    # Download Samanantar
    if args.samanantar or args.essential or args.everything:
        logger.info("\nDownloading Samanantar corpus...")
        downloader = SamanantarDownloader(config)

        if args.all_languages:
            downloader.download_all()
        elif args.languages:
            downloader.download_all(args.languages)
        else:
            # Default: download major languages
            major_langs = ['hi', 'bn', 'ta', 'te', 'mr', 'gu']
            logger.info(f"Downloading major languages: {', '.join(major_langs)}")
            downloader.download_all(major_langs)

    # Download models
    if args.models or args.essential or args.everything:
        logger.info("\nDownloading translation models...")
        downloader = ModelDownloader(config)

        if args.models:
            if 'indictrans2' in args.models:
                downloader.download_indictrans2_models()
            if 'nllb' in args.models:
                downloader.download_nllb_models()
            if any(m in ['llama', 'mistral', 'aya'] for m in args.models):
                llm_models = [m for m in args.models if m in ['llama', 'mistral', 'aya']]
                downloader.download_llm_models(llm_models)

        elif args.everything:
            downloader.download_all(
                include_indictrans2=True,
                include_nllb=True,
                include_llms=True,
                llm_models=['llama', 'mistral', 'aya']
            )
        elif args.essential:
            downloader.download_all(
                include_indictrans2=True,
                include_nllb=True,
                include_llms=False
            )

    if not (args.samanantar or args.models or args.essential or args.everything):
        parser.print_help()
        return

    logger.info("\n" + "="*70)
    logger.info("DOWNLOAD COMPLETE")
    logger.info("="*70)


if __name__ == "__main__":
    main()
