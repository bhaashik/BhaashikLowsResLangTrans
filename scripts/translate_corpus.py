#!/usr/bin/env python3
"""
Translate corpus files (plain text or CoNLL-U format).

This script processes entire directory structures, translating files while
preserving the original directory hierarchy. Useful for Universal Dependencies
corpora and structured datasets.

Features:
- Supports plain text (.txt) and CoNLL-U (.conllu) formats
- Preserves directory structure
- Handles metadata in CoNLL-U files
- Progress tracking and resumable
- Multiple translation strategies

Usage:
    # Translate entire UD corpus
    python scripts/translate_corpus.py \
        --input-dir /path/to/UD_Hindi \
        --output-dir /path/to/UD_Bhojpuri \
        --src hi --tgt bho \
        --format conllu --tiered

    # Translate plain text corpus
    python scripts/translate_corpus.py \
        --input-dir corpus/en \
        --output-dir corpus/hi \
        --src en --tgt hi \
        --format txt --indictrans2
"""

import sys
import os
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
from datetime import datetime
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.translators import IndicTrans2Translator, NLLBTranslator, HindiPivotTranslator
from src.tiered_orchestrator import TieredOrchestrator
from src.utils.config import Config
from src.utils.logger import setup_logger, get_logger


logger = get_logger(__name__)


class CoNLLUParser:
    """Parser for CoNLL-U format files."""

    @staticmethod
    def parse_file(filepath: str) -> List[Dict]:
        """
        Parse CoNLL-U file into sentences with metadata.

        Args:
            filepath: Path to CoNLL-U file

        Returns:
            List of sentence dictionaries with metadata and tokens
        """
        sentences = []
        current_sentence = {
            'metadata': [],
            'tokens': [],
            'text': None
        }

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.rstrip('\n')

                # Empty line = sentence boundary
                if not line:
                    if current_sentence['tokens'] or current_sentence['text']:
                        sentences.append(current_sentence)
                        current_sentence = {
                            'metadata': [],
                            'tokens': [],
                            'text': None
                        }
                    continue

                # Comment/metadata line
                if line.startswith('#'):
                    current_sentence['metadata'].append(line)
                    # Extract text from "# text = ..." line
                    if line.startswith('# text = '):
                        current_sentence['text'] = line[9:]
                    continue

                # Token line
                fields = line.split('\t')
                if len(fields) >= 10:
                    current_sentence['tokens'].append(fields)

        # Add last sentence if exists
        if current_sentence['tokens'] or current_sentence['text']:
            sentences.append(current_sentence)

        return sentences

    @staticmethod
    def write_file(filepath: str, sentences: List[Dict]):
        """
        Write sentences to CoNLL-U file.

        Args:
            filepath: Output file path
            sentences: List of sentence dictionaries
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            for sentence in sentences:
                # Write metadata
                for meta in sentence['metadata']:
                    f.write(meta + '\n')

                # Write tokens
                for token in sentence['tokens']:
                    f.write('\t'.join(token) + '\n')

                # Empty line after sentence
                f.write('\n')


class CorpusTranslator:
    """Translates entire corpus structures preserving directory hierarchy."""

    def __init__(
        self,
        src_lang: str,
        tgt_lang: str,
        translator_type: str = 'indictrans2',
        config: Optional[Config] = None,
        batch_size: int = 32,
        tier_percentages: Optional[Dict] = None
    ):
        """
        Initialize corpus translator.

        Args:
            src_lang: Source language code
            tgt_lang: Target language code
            translator_type: Type of translator ('indictrans2', 'nllb', 'hindi-pivot', 'tiered')
            config: Configuration object
            batch_size: Batch size for translation
            tier_percentages: Custom tier percentages for tiered strategy
        """
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.translator_type = translator_type
        self.config = config or Config()
        self.batch_size = batch_size

        # Initialize translator
        logger.info(f"Initializing {translator_type} translator...")

        if translator_type == 'indictrans2':
            self.translator = IndicTrans2Translator(config=self.config)
        elif translator_type == 'nllb':
            self.translator = NLLBTranslator(config=self.config)
        elif translator_type == 'hindi-pivot':
            self.translator = HindiPivotTranslator(config=self.config)
        elif translator_type == 'tiered':
            self.translator = TieredOrchestrator(
                config=self.config,
                tier_percentages=tier_percentages
            )
        else:
            raise ValueError(f"Unknown translator type: {translator_type}")

        logger.success(f"✓ Translator initialized: {translator_type}")

    def translate_batch(self, texts: List[str]) -> List[str]:
        """
        Translate a batch of texts.

        Args:
            texts: List of texts to translate

        Returns:
            List of translations
        """
        if self.translator_type == 'tiered':
            # Tiered orchestrator returns dict
            results = self.translator.translate(
                texts,
                src_lang=self.src_lang,
                tgt_lang=self.tgt_lang,
                batch_size=self.batch_size,
                show_progress=False,
                save_results=False
            )
            return results['translations']
        else:
            # Regular translators return list
            return self.translator.translate(
                texts,
                src_lang=self.src_lang,
                tgt_lang=self.tgt_lang,
                batch_size=self.batch_size,
                show_progress=False
            )

    def translate_plain_text_file(
        self,
        input_path: Path,
        output_path: Path
    ) -> Dict:
        """
        Translate plain text file (one sentence per line).

        Args:
            input_path: Input file path
            output_path: Output file path

        Returns:
            Dictionary with translation statistics
        """
        logger.info(f"Translating: {input_path}")

        # Read sentences
        with open(input_path, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip()]

        if not sentences:
            logger.warning(f"Empty file: {input_path}")
            output_path.write_text('', encoding='utf-8')
            return {'sentences': 0, 'status': 'empty'}

        # Translate in batches
        translations = []
        for i in range(0, len(sentences), self.batch_size):
            batch = sentences[i:i + self.batch_size]
            batch_trans = self.translate_batch(batch)
            translations.extend(batch_trans)

        # Write output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for trans in translations:
                f.write(trans + '\n')

        logger.success(f"✓ Translated {len(sentences)} sentences")

        return {
            'sentences': len(sentences),
            'status': 'success'
        }

    def translate_conllu_file(
        self,
        input_path: Path,
        output_path: Path
    ) -> Dict:
        """
        Translate CoNLL-U file, preserving structure and metadata.

        Args:
            input_path: Input CoNLL-U file path
            output_path: Output CoNLL-U file path

        Returns:
            Dictionary with translation statistics
        """
        logger.info(f"Translating: {input_path}")

        # Parse input file
        parser = CoNLLUParser()
        sentences = parser.parse_file(str(input_path))

        if not sentences:
            logger.warning(f"Empty file: {input_path}")
            output_path.write_text('', encoding='utf-8')
            return {'sentences': 0, 'status': 'empty'}

        # Extract texts to translate
        texts_to_translate = []
        for sent in sentences:
            # Use "# text = ..." if available, otherwise reconstruct from tokens
            if sent['text']:
                texts_to_translate.append(sent['text'])
            elif sent['tokens']:
                # Reconstruct text from FORM field (column 1)
                words = []
                for token in sent['tokens']:
                    token_id = token[0]
                    # Skip multiword tokens (e.g., "1-2")
                    if '-' not in token_id and '.' not in token_id:
                        words.append(token[1])  # FORM field
                texts_to_translate.append(' '.join(words))
            else:
                texts_to_translate.append('')

        # Translate all texts
        translations = []
        for i in range(0, len(texts_to_translate), self.batch_size):
            batch = texts_to_translate[i:i + self.batch_size]
            batch_trans = self.translate_batch(batch)
            translations.extend(batch_trans)

        # Update sentences with translations
        for sent, trans in zip(sentences, translations):
            # Update "# text = ..." metadata
            new_metadata = []
            text_updated = False

            for meta in sent['metadata']:
                if meta.startswith('# text = '):
                    new_metadata.append(f'# text = {trans}')
                    text_updated = True
                elif meta.startswith('# text_'):
                    # Keep original language metadata
                    # Add translated version
                    new_metadata.append(meta)
                    if not text_updated:
                        new_metadata.append(f'# text_{self.tgt_lang} = {trans}')
                        text_updated = True
                else:
                    new_metadata.append(meta)

            # If no text metadata, add it
            if not text_updated:
                new_metadata.insert(0, f'# text = {trans}')
                if sent['text']:
                    new_metadata.insert(0, f'# text_{self.src_lang} = {sent["text"]}')

            sent['metadata'] = new_metadata
            sent['text'] = trans

        # Write output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        parser.write_file(str(output_path), sentences)

        logger.success(f"✓ Translated {len(sentences)} sentences")

        return {
            'sentences': len(sentences),
            'status': 'success'
        }

    def translate_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        file_format: str = 'txt',
        file_pattern: Optional[str] = None,
        resume: bool = True
    ) -> Dict:
        """
        Translate entire directory structure.

        Args:
            input_dir: Input directory
            output_dir: Output directory
            file_format: File format ('txt' or 'conllu')
            file_pattern: Glob pattern for files (e.g., '*.txt')
            resume: Skip already translated files

        Returns:
            Dictionary with overall statistics
        """
        logger.info("="*70)
        logger.info("CORPUS TRANSLATION")
        logger.info("="*70)
        logger.info(f"Input:  {input_dir}")
        logger.info(f"Output: {output_dir}")
        logger.info(f"Format: {file_format}")
        logger.info(f"Languages: {self.src_lang} → {self.tgt_lang}")
        logger.info("="*70)

        # Determine file extension and pattern
        if file_pattern is None:
            if file_format == 'conllu':
                file_pattern = '**/*.conllu'
            else:
                file_pattern = '**/*.txt'

        # Find all files
        input_files = list(input_dir.glob(file_pattern))

        if not input_files:
            logger.error(f"No files found matching pattern: {file_pattern}")
            return {'files': 0, 'sentences': 0, 'status': 'no_files'}

        logger.info(f"Found {len(input_files)} files to translate")

        # Create metadata file
        metadata = {
            'source_dir': str(input_dir),
            'output_dir': str(output_dir),
            'src_lang': self.src_lang,
            'tgt_lang': self.tgt_lang,
            'translator_type': self.translator_type,
            'file_format': file_format,
            'timestamp': datetime.now().isoformat(),
            'files': []
        }

        # Translate each file
        total_sentences = 0
        successful_files = 0
        failed_files = 0

        for input_path in tqdm(input_files, desc="Translating files"):
            # Determine output path (preserve structure)
            rel_path = input_path.relative_to(input_dir)
            output_path = output_dir / rel_path

            # Skip if already exists and resume=True
            if resume and output_path.exists():
                logger.info(f"Skipping (already exists): {rel_path}")
                continue

            try:
                # Translate based on format
                if file_format == 'conllu':
                    result = self.translate_conllu_file(input_path, output_path)
                else:
                    result = self.translate_plain_text_file(input_path, output_path)

                total_sentences += result['sentences']
                successful_files += 1

                # Add to metadata
                metadata['files'].append({
                    'input': str(rel_path),
                    'output': str(rel_path),
                    'sentences': result['sentences'],
                    'status': result['status']
                })

            except Exception as e:
                logger.error(f"Failed to translate {rel_path}: {str(e)}")
                failed_files += 1

                metadata['files'].append({
                    'input': str(rel_path),
                    'output': str(rel_path),
                    'sentences': 0,
                    'status': 'failed',
                    'error': str(e)
                })

        # Save metadata
        metadata_path = output_dir / 'translation_metadata.json'
        metadata['total_files'] = len(input_files)
        metadata['successful_files'] = successful_files
        metadata['failed_files'] = failed_files
        metadata['total_sentences'] = total_sentences

        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # Print summary
        logger.info("\n" + "="*70)
        logger.info("TRANSLATION COMPLETE")
        logger.info("="*70)
        logger.info(f"Total files: {len(input_files)}")
        logger.info(f"Successful: {successful_files}")
        logger.info(f"Failed: {failed_files}")
        logger.info(f"Total sentences: {total_sentences:,}")
        logger.info(f"Metadata saved: {metadata_path}")
        logger.info("="*70)

        return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Translate corpus files preserving directory structure"
    )

    # Input/Output
    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help="Input directory containing source files"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help="Output directory for translated files"
    )

    # Languages
    parser.add_argument(
        '--src',
        type=str,
        required=True,
        help="Source language code (e.g., en, hi)"
    )
    parser.add_argument(
        '--tgt',
        type=str,
        required=True,
        help="Target language code (e.g., hi, bho)"
    )

    # Format
    parser.add_argument(
        '--format',
        choices=['txt', 'conllu'],
        default='txt',
        help="File format (txt or conllu)"
    )
    parser.add_argument(
        '--pattern',
        type=str,
        help="Glob pattern for files (e.g., '*.txt', 'train/*.conllu')"
    )

    # Translation method
    parser.add_argument(
        '--indictrans2',
        action='store_true',
        help="Use IndicTrans2 translator"
    )
    parser.add_argument(
        '--nllb',
        action='store_true',
        help="Use NLLB translator"
    )
    parser.add_argument(
        '--hindi-pivot',
        action='store_true',
        help="Use Hindi pivot translator"
    )
    parser.add_argument(
        '--tiered',
        action='store_true',
        help="Use tiered strategy (70/20/10)"
    )

    # Tiered strategy options
    parser.add_argument(
        '--tier-free',
        type=float,
        default=70,
        help="Percentage for free tier (default: 70)"
    )
    parser.add_argument(
        '--tier-enhancement',
        type=float,
        default=20,
        help="Percentage for enhancement tier (default: 20)"
    )
    parser.add_argument(
        '--tier-premium',
        type=float,
        default=10,
        help="Percentage for premium tier (default: 10)"
    )

    # Other options
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help="Batch size for translation"
    )
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help="Don't skip already translated files"
    )
    parser.add_argument(
        '--config',
        type=str,
        help="Path to config file"
    )
    parser.add_argument(
        '--log-level',
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

    # Determine translator type
    if args.tiered:
        translator_type = 'tiered'
        tier_percentages = {
            'free': args.tier_free,
            'enhancement': args.tier_enhancement,
            'premium': args.tier_premium
        }
    elif args.nllb:
        translator_type = 'nllb'
        tier_percentages = None
    elif args.hindi_pivot:
        translator_type = 'hindi-pivot'
        tier_percentages = None
    else:
        translator_type = 'indictrans2'
        tier_percentages = None

    # Initialize corpus translator
    corpus_translator = CorpusTranslator(
        src_lang=args.src,
        tgt_lang=args.tgt,
        translator_type=translator_type,
        config=config,
        batch_size=args.batch_size,
        tier_percentages=tier_percentages
    )

    # Translate directory
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)

    corpus_translator.translate_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        file_format=args.format,
        file_pattern=args.pattern,
        resume=not args.no_resume
    )


if __name__ == "__main__":
    main()
