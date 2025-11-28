"""Downloader for translation models."""

import os
from pathlib import Path
from typing import List, Optional, Dict
from huggingface_hub import snapshot_download, hf_hub_download
from tqdm import tqdm

from ..utils.config import Config
from ..utils.logger import get_logger


logger = get_logger(__name__)


class ModelDownloader:
    """Downloads and manages translation models from HuggingFace."""

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize model downloader.

        Args:
            config: Configuration object. If None, creates a new one.
        """
        self.config = config or Config()
        self.cache_dir = self.config.transformers_cache
        self.hf_token = self.config.hf_token

        # Enable faster downloads
        os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

        logger.info("Initialized model downloader")
        logger.info(f"Cache directory: {self.cache_dir}")

    def download_model(
        self,
        model_id: str,
        force_download: bool = False,
        max_retries: int = 3
    ) -> Optional[str]:
        """
        Download a model from HuggingFace Hub.

        Args:
            model_id: HuggingFace model ID (e.g., 'ai4bharat/indictrans2-en-indic-1B')
            force_download: Force re-download even if cached
            max_retries: Maximum number of download attempts

        Returns:
            Path to downloaded model or None if failed
        """
        logger.info(f"{'='*70}")
        logger.info(f"Downloading model: {model_id}")
        logger.info(f"{'='*70}")

        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Attempt {attempt}/{max_retries}")

                path = snapshot_download(
                    repo_id=model_id,
                    cache_dir=self.cache_dir,
                    token=self.hf_token,
                    resume_download=True,
                    force_download=force_download
                )

                logger.success(f"✓ Downloaded: {model_id}")
                logger.info(f"Path: {path}")
                return path

            except Exception as e:
                logger.error(f"Attempt {attempt} failed: {str(e)}")
                if attempt < max_retries:
                    logger.info("Retrying in 10 seconds...")
                    import time
                    time.sleep(10)
                else:
                    logger.error(f"✗ Failed after {max_retries} attempts: {model_id}")
                    return None

    def download_indictrans2_models(
        self,
        variants: Optional[List[str]] = None
    ) -> Dict[str, Optional[str]]:
        """
        Download IndicTrans2 models.

        Args:
            variants: List of variants to download. If None, downloads all.
                     Options: 'en_to_indic_1b', 'indic_to_en_1b', 'indic_to_indic_1b',
                             'en_to_indic_200m', 'indic_to_en_200m'

        Returns:
            Dictionary mapping variant names to download paths
        """
        logger.info("="*70)
        logger.info("DOWNLOADING INDICTRANS2 MODELS")
        logger.info("="*70)

        all_variants = {
            'en_to_indic_1b': self.config.get_model_path('indictrans2', 'en_to_indic_1b'),
            'indic_to_en_1b': self.config.get_model_path('indictrans2', 'indic_to_en_1b'),
            'indic_to_indic_1b': self.config.get_model_path('indictrans2', 'indic_to_indic_1b'),
            'en_to_indic_200m': self.config.get_model_path('indictrans2', 'en_to_indic_200m'),
            'indic_to_en_200m': self.config.get_model_path('indictrans2', 'indic_to_en_200m'),
        }

        if variants:
            models_to_download = {k: v for k, v in all_variants.items() if k in variants}
        else:
            models_to_download = all_variants

        results = {}
        for variant, model_id in models_to_download.items():
            logger.info(f"\nDownloading {variant}...")
            path = self.download_model(model_id)
            results[variant] = path

        logger.info("\n" + "="*70)
        logger.info("INDICTRANS2 DOWNLOAD COMPLETE")
        logger.info("="*70)

        successful = sum(1 for p in results.values() if p is not None)
        logger.info(f"Successful: {successful}/{len(results)}")

        return results

    def download_nllb_models(
        self,
        variants: Optional[List[str]] = None
    ) -> Dict[str, Optional[str]]:
        """
        Download NLLB-200 models.

        Args:
            variants: List of variants to download. If None, downloads default.
                     Options: 'distilled_600m', 'base_1_3b', 'large_3_3b'

        Returns:
            Dictionary mapping variant names to download paths
        """
        logger.info("="*70)
        logger.info("DOWNLOADING NLLB-200 MODELS")
        logger.info("="*70)

        all_variants = {
            'distilled_600m': self.config.get_model_path('nllb', 'distilled_600m'),
            'base_1_3b': self.config.get_model_path('nllb', 'base_1_3b'),
            'large_3_3b': self.config.get_model_path('nllb', 'large_3_3b'),
        }

        if variants:
            models_to_download = {k: v for k, v in all_variants.items() if k in variants}
        else:
            # Default: download only distilled version
            models_to_download = {'distilled_600m': all_variants['distilled_600m']}

        results = {}
        for variant, model_id in models_to_download.items():
            logger.info(f"\nDownloading {variant}...")
            path = self.download_model(model_id)
            results[variant] = path

        logger.info("\n" + "="*70)
        logger.info("NLLB-200 DOWNLOAD COMPLETE")
        logger.info("="*70)

        successful = sum(1 for p in results.values() if p is not None)
        logger.info(f"Successful: {successful}/{len(results)}")

        return results

    def download_llm_models(
        self,
        models: Optional[List[str]] = None
    ) -> Dict[str, Optional[str]]:
        """
        Download LLM models for translation.

        Args:
            models: List of models to download. If None, downloads none.
                   Options: 'llama', 'mistral', 'aya'

        Returns:
            Dictionary mapping model names to download paths
        """
        if not models:
            logger.warning("No LLM models specified. Skipping...")
            return {}

        logger.info("="*70)
        logger.info("DOWNLOADING LLM MODELS")
        logger.info("="*70)
        logger.warning("Note: LLMs are large (7B-8B parameters). This will take time.")

        all_models = {
            'llama': self.config.get_model_path('llms', 'llama'),
            'mistral': self.config.get_model_path('llms', 'mistral'),
            'aya': self.config.get_model_path('llms', 'aya'),
        }

        models_to_download = {k: v for k, v in all_models.items() if k in models}

        results = {}
        for model_name, model_id in models_to_download.items():
            logger.info(f"\nDownloading {model_name}...")
            path = self.download_model(model_id)
            results[model_name] = path

        logger.info("\n" + "="*70)
        logger.info("LLM DOWNLOAD COMPLETE")
        logger.info("="*70)

        successful = sum(1 for p in results.values() if p is not None)
        logger.info(f"Successful: {successful}/{len(results)}")

        return results

    def download_all(
        self,
        include_indictrans2: bool = True,
        include_nllb: bool = True,
        include_llms: bool = False,
        indictrans2_variants: Optional[List[str]] = None,
        nllb_variants: Optional[List[str]] = None,
        llm_models: Optional[List[str]] = None
    ) -> Dict:
        """
        Download all specified models.

        Args:
            include_indictrans2: Whether to download IndicTrans2 models
            include_nllb: Whether to download NLLB models
            include_llms: Whether to download LLM models
            indictrans2_variants: Specific IndicTrans2 variants
            nllb_variants: Specific NLLB variants
            llm_models: Specific LLM models

        Returns:
            Dictionary with all download results
        """
        results = {
            'indictrans2': {},
            'nllb': {},
            'llms': {}
        }

        if include_indictrans2:
            results['indictrans2'] = self.download_indictrans2_models(indictrans2_variants)

        if include_nllb:
            results['nllb'] = self.download_nllb_models(nllb_variants)

        if include_llms:
            results['llms'] = self.download_llm_models(llm_models)

        return results


def main():
    """CLI interface for model downloader."""
    import argparse

    parser = argparse.ArgumentParser(description="Download translation models")
    parser.add_argument(
        "--indictrans2",
        action="store_true",
        help="Download IndicTrans2 models"
    )
    parser.add_argument(
        "--nllb",
        action="store_true",
        help="Download NLLB-200 models"
    )
    parser.add_argument(
        "--llms",
        nargs="+",
        choices=['llama', 'mistral', 'aya'],
        help="Download specific LLM models"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all essential models (IndicTrans2 + NLLB)"
    )

    args = parser.parse_args()

    # Set up logger
    from ..utils.logger import setup_logger
    setup_logger(level="INFO")

    downloader = ModelDownloader()

    if args.all:
        downloader.download_all(include_indictrans2=True, include_nllb=True, include_llms=False)
    else:
        if args.indictrans2:
            downloader.download_indictrans2_models()
        if args.nllb:
            downloader.download_nllb_models()
        if args.llms:
            downloader.download_llm_models(args.llms)

        if not (args.indictrans2 or args.nllb or args.llms):
            parser.print_help()


if __name__ == "__main__":
    main()
