"""Downloader for Samanantar parallel corpus."""

import sys
from pathlib import Path
from typing import List, Optional, Dict
from datasets import load_dataset
from tqdm import tqdm

from ..utils.config import Config
from ..utils.logger import get_logger


logger = get_logger(__name__)


class SamanantarDownloader:
    """Downloads and manages Samanantar parallel corpus."""

    # All available languages in Samanantar
    ALL_LANGUAGES = ['as', 'bn', 'gu', 'hi', 'kn', 'ml', 'mr', 'or', 'pa', 'ta', 'te']

    LANGUAGE_NAMES = {
        'as': 'Assamese',
        'bn': 'Bengali',
        'gu': 'Gujarati',
        'hi': 'Hindi',
        'kn': 'Kannada',
        'ml': 'Malayalam',
        'mr': 'Marathi',
        'or': 'Odia',
        'pa': 'Punjabi',
        'ta': 'Tamil',
        'te': 'Telugu'
    }

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize Samanantar downloader.

        Args:
            config: Configuration object. If None, creates a new one.
        """
        self.config = config or Config()
        self.dataset_id = self.config.get("data_sources.samanantar.dataset_id", "ai4bharat/samanantar")
        self.cache_dir = self.config.hf_datasets_cache

        logger.info(f"Initialized Samanantar downloader")
        logger.info(f"Dataset ID: {self.dataset_id}")
        logger.info(f"Cache directory: {self.cache_dir}")

    def download_language(self, language: str, force_download: bool = False) -> Optional[Dict]:
        """
        Download a single language pair.

        Args:
            language: Language code (e.g., 'hi', 'bn')
            force_download: Force re-download even if cached

        Returns:
            Dictionary with download information or None if failed
        """
        if language not in self.ALL_LANGUAGES:
            logger.error(f"Invalid language code: {language}")
            logger.info(f"Valid codes: {', '.join(self.ALL_LANGUAGES)}")
            return None

        lang_name = self.LANGUAGE_NAMES.get(language, language.upper())
        logger.info(f"{'='*70}")
        logger.info(f"Downloading: English ↔ {lang_name} ({language})")
        logger.info(f"{'='*70}")

        try:
            # Download dataset
            dataset = load_dataset(
                self.dataset_id,
                language,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )

            num_pairs = len(dataset['train'])
            logger.success(f"✓ Downloaded: {num_pairs:,} sentence pairs")

            # Show sample
            if num_pairs > 0:
                sample = dataset['train'][0]
                logger.info(f"\nSample:")
                logger.info(f"  EN: {sample['src'][:100]}...")
                logger.info(f"  {language.upper()}: {sample['tgt'][:100]}...")

            return {
                "language": language,
                "language_name": lang_name,
                "num_pairs": num_pairs,
                "dataset": dataset,
                "status": "success"
            }

        except Exception as e:
            logger.error(f"✗ Failed to download {lang_name}: {str(e)}")
            return {
                "language": language,
                "language_name": lang_name,
                "num_pairs": 0,
                "dataset": None,
                "status": "failed",
                "error": str(e)
            }

    def download_all(self, languages: Optional[List[str]] = None) -> Dict:
        """
        Download multiple language pairs.

        Args:
            languages: List of language codes. If None, downloads all.

        Returns:
            Dictionary with summary of downloads
        """
        if languages is None:
            languages = self.ALL_LANGUAGES

        logger.info("="*70)
        logger.info("SAMANANTAR CORPUS DOWNLOAD")
        logger.info("="*70)
        logger.info(f"Languages to download: {len(languages)}")
        logger.info(f"Cache directory: {self.cache_dir}")
        logger.info("="*70)

        results = []
        total_pairs = 0
        successful = []
        failed = []

        for lang in tqdm(languages, desc="Downloading languages"):
            result = self.download_language(lang)
            results.append(result)

            if result and result["status"] == "success":
                successful.append(lang)
                total_pairs += result["num_pairs"]
            else:
                failed.append(lang)

        # Print summary
        logger.info("\n" + "="*70)
        logger.info("DOWNLOAD SUMMARY")
        logger.info("="*70)
        logger.info(f"Total sentence pairs: {total_pairs:,}")
        logger.info(f"Successful: {len(successful)}/{len(languages)}")

        if successful:
            logger.success(f"Success: {', '.join(successful)}")

        if failed:
            logger.error(f"Failed: {', '.join(failed)}")

        logger.info("="*70)

        return {
            "total_pairs": total_pairs,
            "successful": successful,
            "failed": failed,
            "results": results
        }

    def verify_download(self, language: str) -> bool:
        """
        Verify that a language has been downloaded successfully.

        Args:
            language: Language code

        Returns:
            True if dataset is available, False otherwise
        """
        try:
            dataset = load_dataset(
                self.dataset_id,
                language,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            return len(dataset['train']) > 0
        except Exception as e:
            logger.error(f"Verification failed for {language}: {str(e)}")
            return False

    def get_dataset(self, language: str):
        """
        Load a previously downloaded dataset.

        Args:
            language: Language code

        Returns:
            HuggingFace Dataset object

        Raises:
            ValueError: If language is invalid or not downloaded
        """
        if language not in self.ALL_LANGUAGES:
            raise ValueError(f"Invalid language code: {language}")

        try:
            dataset = load_dataset(
                self.dataset_id,
                language,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            return dataset
        except Exception as e:
            raise ValueError(f"Failed to load dataset for {language}: {str(e)}")


def main():
    """CLI interface for Samanantar downloader."""
    import argparse

    parser = argparse.ArgumentParser(description="Download Samanantar parallel corpus")
    parser.add_argument(
        "--languages",
        nargs="+",
        help=f"Language codes to download. Options: {', '.join(SamanantarDownloader.ALL_LANGUAGES)}"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all available languages"
    )
    parser.add_argument(
        "--verify",
        nargs="+",
        help="Verify downloaded languages"
    )

    args = parser.parse_args()

    # Set up logger
    from ..utils.logger import setup_logger
    setup_logger(level="INFO")

    downloader = SamanantarDownloader()

    if args.verify:
        logger.info("Verifying downloads...")
        for lang in args.verify:
            status = downloader.verify_download(lang)
            if status:
                logger.success(f"✓ {lang}: Available")
            else:
                logger.error(f"✗ {lang}: Not available")

    elif args.all:
        downloader.download_all()

    elif args.languages:
        downloader.download_all(args.languages)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
