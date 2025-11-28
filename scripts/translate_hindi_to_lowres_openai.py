#!/usr/bin/env python3
"""
Translate Hindi plain-text files to low-resource languages using OpenAI with prompt caching.

This script:
1. Uses monolingual examples from PV-BMM-Public-Data for prompt caching
2. Translates Hindi files in input/converted/Hindi/plain-text/by_file/
3. Recreates directory structure in output directory
4. Supports Bhojpuri, Magahi, and Maithili

Usage:
    python scripts/translate_hindi_to_lowres_openai.py --target-lang bho
    python scripts/translate_hindi_to_lowres_openai.py --target-lang mag --dry-run
    python scripts/translate_hindi_to_lowres_openai.py --target-lang mai --max-files 5
"""

import sys
import argparse
from pathlib import Path
from typing import List, Dict
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from universal_translate import TranslationRequest, TranslationUnit
from universal_translate.providers.openai_provider import OpenAIProvider
from universal_translate.prompts import PromptManager


class DirectoryTranslator:
    """Handles translation of directory structures."""

    def __init__(
        self,
        provider: OpenAIProvider,
        input_base: Path,
        output_base: Path,
        target_lang: str,
        src_lang: str = 'hi'
    ):
        """
        Initialize directory translator.

        Args:
            provider: Translation provider
            input_base: Base input directory
            output_base: Base output directory
            target_lang: Target language code
            src_lang: Source language code
        """
        self.provider = provider
        self.input_base = input_base
        self.output_base = output_base
        self.target_lang = target_lang
        self.src_lang = src_lang

        self.stats = {
            'files_processed': 0,
            'sentences_translated': 0,
            'total_cost': 0.0,
            'files_skipped': 0,
            'errors': []
        }

    def translate_directory(
        self,
        max_files: int = None,
        dry_run: bool = False,
        batch_size: int = 10
    ):
        """
        Translate all files in directory structure.

        Args:
            max_files: Maximum files to process (for testing)
            dry_run: If True, only show what would be done
            batch_size: Number of sentences to translate at once
        """
        print("="*70)
        print("DIRECTORY TRANSLATION")
        print("="*70)
        print(f"Input:  {self.input_base}")
        print(f"Output: {self.output_base}")
        print(f"Languages: {self.src_lang} → {self.target_lang}")
        print(f"Dry run: {dry_run}")
        print("="*70 + "\n")

        # Find all .txt files
        txt_files = sorted(self.input_base.rglob('*.txt'))

        if not txt_files:
            print(f"✗ No .txt files found in {self.input_base}")
            return

        print(f"Found {len(txt_files)} text files")

        if max_files:
            txt_files = txt_files[:max_files]
            print(f"Processing first {max_files} files (--max-files limit)")

        # Process each file
        for i, input_file in enumerate(txt_files, 1):
            print(f"\n[{i}/{len(txt_files)}] {input_file.relative_to(self.input_base)}")

            # Calculate output path (preserve directory structure)
            rel_path = input_file.relative_to(self.input_base)
            output_file = self.output_base / rel_path

            if dry_run:
                print(f"  Would translate: {input_file}")
                print(f"  Would write to:  {output_file}")
                continue

            # Translate file
            try:
                self._translate_file(input_file, output_file, batch_size)
                self.stats['files_processed'] += 1
            except Exception as e:
                print(f"  ✗ Error: {e}")
                self.stats['errors'].append({
                    'file': str(input_file),
                    'error': str(e)
                })

        # Print summary
        self._print_summary()

    def _translate_file(self, input_file: Path, output_file: Path, batch_size: int):
        """Translate a single file."""
        # Read source sentences
        with open(input_file, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip()]

        if not sentences:
            print(f"  ⚠ Empty file, skipping")
            self.stats['files_skipped'] += 1
            return

        print(f"  Sentences: {len(sentences)}")

        # Create output directory
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Translate in batches
        all_translations = []

        for batch_start in range(0, len(sentences), batch_size):
            batch_end = min(batch_start + batch_size, len(sentences))
            batch = sentences[batch_start:batch_end]

            print(f"  Translating batch {batch_start+1}-{batch_end}/{len(sentences)}...", end=' ')

            # Create translation request
            units = [
                TranslationUnit(text=text, index=i)
                for i, text in enumerate(batch, start=batch_start)
            ]

            request = TranslationRequest(
                units=units,
                src_lang=self.src_lang,
                tgt_lang=self.target_lang
            )

            # Translate
            with self.provider:
                response = self.provider.translate_sync(request)

            # Extract translations
            for result in sorted(response.results, key=lambda x: x.index):
                all_translations.append(result.translation)

            print(f"Done (cost: ${response.total_cost:.4f})")
            self.stats['total_cost'] += response.total_cost
            self.stats['sentences_translated'] += len(batch)

        # Write translations
        with open(output_file, 'w', encoding='utf-8') as f:
            for translation in all_translations:
                f.write(translation + '\n')

        print(f"  ✓ Written: {output_file}")

    def _print_summary(self):
        """Print translation summary."""
        print("\n" + "="*70)
        print("TRANSLATION SUMMARY")
        print("="*70)
        print(f"Files processed:       {self.stats['files_processed']}")
        print(f"Files skipped:         {self.stats['files_skipped']}")
        print(f"Sentences translated:  {self.stats['sentences_translated']:,}")
        print(f"Total cost:            ${self.stats['total_cost']:.2f} USD")

        if self.stats['sentences_translated'] > 0:
            avg_cost = self.stats['total_cost'] / self.stats['sentences_translated']
            print(f"Average cost/sentence: ${avg_cost:.6f} USD")

        if self.stats['errors']:
            print(f"\nErrors: {len(self.stats['errors'])}")
            for error in self.stats['errors'][:5]:  # Show first 5
                print(f"  - {error['file']}: {error['error']}")

        print("="*70)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Translate Hindi to low-resource languages using OpenAI with prompt caching"
    )

    parser.add_argument(
        '--target-lang',
        required=True,
        choices=['bho', 'mag', 'mai'],
        help="Target language (bho=Bhojpuri, mag=Magahi, mai=Maithili)"
    )

    parser.add_argument(
        '--model',
        default='gpt-4o-mini',
        help="OpenAI model to use (default: gpt-4o-mini)"
    )

    parser.add_argument(
        '--input-dir',
        type=str,
        default='input/converted/Hindi/plain-text/by_file',
        help="Input directory with Hindi files (default: input/converted/Hindi/plain-text/by_file)"
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        help="Output directory (default: output/{Language}/plain-text/by_file)"
    )

    parser.add_argument(
        '--max-files',
        type=int,
        help="Maximum number of files to translate (for testing)"
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=10,
        help="Number of sentences per API call (default: 10)"
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="Show what would be done without translating"
    )

    parser.add_argument(
        '--estimate-only',
        action='store_true',
        help="Only estimate cost, don't translate"
    )

    args = parser.parse_args()

    # Set paths
    base_dir = Path(__file__).parent.parent
    input_base = base_dir / args.input_dir

    if not input_base.exists():
        print(f"✗ Input directory not found: {input_base}")
        return 1

    # Language names
    lang_names = {
        'bho': 'Bhojpuri',
        'mag': 'Magahi',
        'mai': 'Maithili'
    }

    target_lang_name = lang_names[args.target_lang]

    # Output directory
    if args.output_dir:
        output_base = base_dir / args.output_dir
    else:
        output_base = base_dir / 'output' / target_lang_name / 'plain-text' / 'by_file'

    # Load prompt configuration
    prompt_config = base_dir / f"universal_translate/config/prompts/hi_to_{args.target_lang}_openai.yaml"

    if not prompt_config.exists():
        print(f"✗ Prompt config not found: {prompt_config}")
        return 1

    print("\nLoading configuration...")
    prompt_mgr = PromptManager(str(prompt_config))
    print(f"  Prompt: {prompt_mgr.config.get('name')}")
    print(f"  Examples: {len(prompt_mgr.examples)} monolingual sentences")
    print(f"  Caching: {prompt_mgr.supports_caching()}")

    # Create provider
    print(f"\nInitializing OpenAI provider ({args.model})...")
    provider = OpenAIProvider(
        model=args.model,
        prompt_manager=prompt_mgr,
        use_caching=True
    )

    # Cost estimation
    if args.estimate_only or args.dry_run:
        print("\nEstimating costs...")

        # Count sentences
        txt_files = list(input_base.rglob('*.txt'))
        if args.max_files:
            txt_files = txt_files[:args.max_files]

        total_sentences = 0
        total_chars = 0

        for txt_file in txt_files:
            with open(txt_file, 'r', encoding='utf-8') as f:
                sentences = [line.strip() for line in f if line.strip()]
                total_sentences += len(sentences)
                total_chars += sum(len(s) for s in sentences)

        # Create sample request for estimation
        units = [TranslationUnit(text="x" * (total_chars // total_sentences if total_sentences > 0 else 100), index=0)] * total_sentences

        request = TranslationRequest(units=units, src_lang='hi', tgt_lang=args.target_lang)

        estimate = provider.get_cost_estimate(request)

        print(f"\nCost Estimate:")
        print(f"  Files: {len(txt_files)}")
        print(f"  Sentences: {total_sentences:,}")
        print(f"  Estimated input cost: ${estimate.input_cost:.2f}")
        print(f"  Estimated output cost: ${estimate.output_cost:.2f}")
        print(f"  Total estimated cost: ${estimate.total_cost:.2f}")
        print(f"  Cost per sentence: ${estimate.total_cost/total_sentences:.6f}")

        print(f"\nWith caching benefits:")
        print(f"  System tokens: ~{estimate.metadata.get('estimated_system_tokens', 0)} tokens")
        print(f"  First request: Full cost")
        print(f"  Subsequent: ~50% savings on system prompt")

        if args.estimate_only:
            return 0

    # Create translator
    translator = DirectoryTranslator(
        provider=provider,
        input_base=input_base,
        output_base=output_base,
        target_lang=args.target_lang
    )

    # Translate
    translator.translate_directory(
        max_files=args.max_files,
        dry_run=args.dry_run,
        batch_size=args.batch_size
    )

    return 0


if __name__ == '__main__':
    sys.exit(main())
