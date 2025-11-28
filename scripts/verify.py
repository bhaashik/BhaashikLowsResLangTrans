#!/usr/bin/env python3
"""
Verification script to check downloads and system setup.

Usage:
    python scripts/verify.py --all
    python scripts/verify.py --samanantar --languages hi bn ta
    python scripts/verify.py --models
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


def check_environment():
    """Check environment setup."""
    logger.info("\n" + "="*70)
    logger.info("ENVIRONMENT CHECK")
    logger.info("="*70)

    # Check Python packages
    packages = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'datasets': 'Datasets',
        'anthropic': 'Anthropic',
        'sacrebleu': 'SacreBLEU'
    }

    all_ok = True
    for package, name in packages.items():
        try:
            __import__(package)
            logger.success(f"✓ {name}")
        except ImportError:
            logger.error(f"✗ {name} not installed")
            all_ok = False

    # Check CUDA
    try:
        import torch
        if torch.cuda.is_available():
            logger.success(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("⚠ CUDA not available (CPU only)")
    except:
        pass

    return all_ok


def check_directories(config: Config):
    """Check directory structure."""
    logger.info("\n" + "="*70)
    logger.info("DIRECTORY CHECK")
    logger.info("="*70)

    paths = config.get_paths()

    for name, path in paths.items():
        if path.exists():
            # Get size
            try:
                total_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                size_gb = total_size / (1024**3)
                logger.success(f"✓ {name}: {path} ({size_gb:.2f} GB)")
            except:
                logger.success(f"✓ {name}: {path}")
        else:
            logger.warning(f"⚠ {name}: {path} (does not exist)")


def check_samanantar(config: Config, languages: list = None):
    """Check Samanantar downloads."""
    logger.info("\n" + "="*70)
    logger.info("SAMANANTAR CHECK")
    logger.info("="*70)

    downloader = SamanantarDownloader(config)

    if languages is None:
        languages = downloader.ALL_LANGUAGES

    results = []
    for lang in languages:
        status = downloader.verify_download(lang)
        results.append((lang, status))
        if status:
            logger.success(f"✓ {lang}: Available")
        else:
            logger.error(f"✗ {lang}: Not available")

    successful = sum(1 for _, status in results if status)
    logger.info(f"\nAvailable: {successful}/{len(languages)}")

    return all(status for _, status in results)


def check_models(config: Config):
    """Check model downloads."""
    logger.info("\n" + "="*70)
    logger.info("MODEL CHECK")
    logger.info("="*70)

    # Check IndicTrans2
    logger.info("\nIndicTrans2 Models:")
    indictrans2_models = [
        ('en_to_indic_200m', config.get_model_path('indictrans2', 'en_to_indic_200m')),
        ('indic_to_en_200m', config.get_model_path('indictrans2', 'indic_to_en_200m')),
    ]

    all_ok = True
    for name, model_path in indictrans2_models:
        try:
            from transformers import AutoTokenizer
            AutoTokenizer.from_pretrained(
                model_path,
                cache_dir=config.transformers_cache,
                trust_remote_code=True
            )
            logger.success(f"  ✓ {name}")
        except Exception as e:
            logger.error(f"  ✗ {name}: {str(e)[:50]}")
            all_ok = False

    # Check NLLB
    logger.info("\nNLLB Models:")
    nllb_model = config.get_model_path('nllb', 'default')
    try:
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(
            nllb_model,
            cache_dir=config.transformers_cache
        )
        logger.success(f"  ✓ NLLB distilled 600M")
    except Exception as e:
        logger.error(f"  ✗ NLLB: {str(e)[:50]}")
        all_ok = False

    return all_ok


def check_api_keys(config: Config):
    """Check API keys."""
    logger.info("\n" + "="*70)
    logger.info("API KEYS CHECK")
    logger.info("="*70)

    keys = {
        'HuggingFace': config.hf_token,
        'Anthropic': config.anthropic_api_key,
        'OpenAI': config.openai_api_key,
        'Google': config.google_credentials,
        'Azure': config.azure_key
    }

    for name, key in keys.items():
        if key:
            masked = key[:4] + '...' + key[-4:] if len(key) > 8 else '***'
            logger.success(f"✓ {name}: {masked}")
        else:
            logger.warning(f"⚠ {name}: Not configured")


def main():
    parser = argparse.ArgumentParser(
        description="Verify BhaashikLowsResLangTrans setup"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all checks"
    )
    parser.add_argument(
        "--environment",
        action="store_true",
        help="Check environment and packages"
    )
    parser.add_argument(
        "--directories",
        action="store_true",
        help="Check directory structure"
    )
    parser.add_argument(
        "--samanantar",
        action="store_true",
        help="Check Samanantar downloads"
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        help="Specific languages to check"
    )
    parser.add_argument(
        "--models",
        action="store_true",
        help="Check model downloads"
    )
    parser.add_argument(
        "--api-keys",
        action="store_true",
        help="Check API keys"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config file"
    )

    args = parser.parse_args()

    # Initialize
    config = Config(args.config) if args.config else Config()

    setup_logger(level='INFO', console=True)

    logger.info("="*70)
    logger.info("BHAASHIK LOW-RESOURCE LANGUAGE TRANSLATION")
    logger.info("VERIFICATION SCRIPT")
    logger.info("="*70)

    # Run checks
    if args.all or args.environment:
        check_environment()

    if args.all or args.directories:
        check_directories(config)

    if args.all or args.samanantar:
        check_samanantar(config, args.languages)

    if args.all or args.models:
        check_models(config)

    if args.all or args.api_keys:
        check_api_keys(config)

    if not any([args.all, args.environment, args.directories,
                args.samanantar, args.models, args.api_keys]):
        parser.print_help()

    logger.info("\n" + "="*70)
    logger.info("VERIFICATION COMPLETE")
    logger.info("="*70)


if __name__ == "__main__":
    main()
