#!/usr/bin/env python3
"""
Extract plain text from CoNLL-U files while maintaining directory structure.

This script:
1. Reads CoNLL-U files from input/converted directory
2. Extracts plain text sentences (from tokens, not from # text = lines which may be empty)
3. Creates plain-text directory with identical structure
4. Generates JSON mappings from sent_id to sentence position (1-based index)
5. Creates separate mapping files for: by_file, by_domain, and merged

Usage:
    python scripts/extract_plaintext_from_conllu.py
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
import conllu


# Simple logger class
class SimpleLogger:
    def info(self, msg):
        print(f"[INFO] {msg}")

    def error(self, msg):
        print(f"[ERROR] {msg}", file=sys.stderr)


logger = SimpleLogger()


def extract_text_from_sentence(sentence: conllu.models.TokenList) -> str:
    """
    Extract plain text from a CoNLL-U sentence.

    Args:
        sentence: CoNLL-U TokenList

    Returns:
        Plain text string constructed from tokens
    """
    # Try to get from metadata first
    if sentence.metadata and 'text' in sentence.metadata:
        text = sentence.metadata['text']
        if text and text.strip():
            return text.strip()

    # Fallback: reconstruct from tokens
    tokens = []
    for token in sentence:
        # Skip multiword tokens (those with range IDs like "1-2")
        if isinstance(token['id'], int):
            tokens.append(token['form'])

    return ' '.join(tokens)


def get_sent_id(sentence: conllu.models.TokenList) -> str:
    """
    Extract sent_id from sentence metadata.

    Args:
        sentence: CoNLL-U TokenList

    Returns:
        sent_id string, or generated ID if not found
    """
    if sentence.metadata and 'sent_id' in sentence.metadata:
        return sentence.metadata['sent_id']
    return f"sentence_{id(sentence)}"


def process_conllu_file(
    input_path: Path,
    output_path: Path,
    relative_path: str
) -> Tuple[List[str], Dict[str, int]]:
    """
    Process a single CoNLL-U file.

    Args:
        input_path: Path to input CoNLL-U file
        output_path: Path to output plain text file
        relative_path: Relative path for logging

    Returns:
        Tuple of (plain_texts, sent_id_mapping)
    """
    logger.info(f"Processing: {relative_path}")

    try:
        # Read CoNLL-U file
        with open(input_path, 'r', encoding='utf-8') as f:
            sentences = conllu.parse(f.read())

        plain_texts = []
        sent_id_mapping = {}

        for idx, sentence in enumerate(sentences, start=1):
            # Extract text
            text = extract_text_from_sentence(sentence)
            plain_texts.append(text)

            # Create mapping: sent_id -> position (1-based)
            sent_id = get_sent_id(sentence)
            sent_id_mapping[sent_id] = idx

        # Write plain text file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for text in plain_texts:
                f.write(text + '\n')

        logger.info(f"  ✓ Extracted {len(plain_texts)} sentences -> {output_path.relative_to(output_path.parents[3])}")

        return plain_texts, sent_id_mapping

    except Exception as e:
        logger.error(f"  ✗ Error processing {relative_path}: {e}")
        return [], {}


def process_language_directory(
    language: str,
    input_base: Path,
    output_base: Path
) -> Dict[str, any]:
    """
    Process all CoNLL-U files for a language.

    Args:
        language: Language name (e.g., 'Hindi')
        input_base: Base input directory
        output_base: Base output directory

    Returns:
        Dictionary with statistics and mappings
    """
    logger.info("="*70)
    logger.info(f"Processing {language}")
    logger.info("="*70)

    lang_input_dir = input_base / language
    lang_output_dir = output_base / language

    # Create output directories
    (lang_output_dir / 'plain-text' / 'by_domain').mkdir(parents=True, exist_ok=True)
    (lang_output_dir / 'plain-text' / 'by_file').mkdir(parents=True, exist_ok=True)
    (lang_output_dir / 'plain-text' / 'merged').mkdir(parents=True, exist_ok=True)
    (lang_output_dir / 'mappings').mkdir(parents=True, exist_ok=True)

    all_mappings = {
        'by_domain': {},
        'by_file': {},
        'merged': {}
    }

    statistics = {
        'language': language,
        'total_files': 0,
        'total_sentences': 0,
        'by_domain': {},
        'by_file': {},
        'merged': {}
    }

    # Process by_domain files
    by_domain_dir = lang_input_dir / 'by_domain'
    if by_domain_dir.exists():
        logger.info("\nProcessing by_domain files...")
        for conllu_file in sorted(by_domain_dir.glob('*.conllu')):
            relative_path = f"{language}/by_domain/{conllu_file.name}"
            output_file = lang_output_dir / 'plain-text' / 'by_domain' / conllu_file.name.replace('.conllu', '.txt')

            texts, mapping = process_conllu_file(conllu_file, output_file, relative_path)

            if mapping:
                domain_name = conllu_file.stem
                all_mappings['by_domain'][domain_name] = mapping
                statistics['by_domain'][domain_name] = len(texts)
                statistics['total_files'] += 1
                statistics['total_sentences'] += len(texts)

    # Process by_file subdirectories
    by_file_dir = lang_input_dir / 'by_file'
    if by_file_dir.exists():
        logger.info("\nProcessing by_file subdirectories...")
        for domain_dir in sorted(by_file_dir.iterdir()):
            if not domain_dir.is_dir():
                continue

            domain_name = domain_dir.name
            output_domain_dir = lang_output_dir / 'plain-text' / 'by_file' / domain_name
            output_domain_dir.mkdir(parents=True, exist_ok=True)

            all_mappings['by_file'][domain_name] = {}
            statistics['by_file'][domain_name] = {}

            for conllu_file in sorted(domain_dir.glob('*.conllu')):
                relative_path = f"{language}/by_file/{domain_name}/{conllu_file.name}"
                output_file = output_domain_dir / conllu_file.name.replace('.conllu', '.txt')

                texts, mapping = process_conllu_file(conllu_file, output_file, relative_path)

                if mapping:
                    file_name = conllu_file.stem
                    all_mappings['by_file'][domain_name][file_name] = mapping
                    statistics['by_file'][domain_name][file_name] = len(texts)
                    statistics['total_files'] += 1
                    statistics['total_sentences'] += len(texts)

    # Process merged files
    merged_dir = lang_input_dir / 'merged'
    if merged_dir.exists():
        logger.info("\nProcessing merged files...")
        for conllu_file in sorted(merged_dir.glob('*.conllu')):
            relative_path = f"{language}/merged/{conllu_file.name}"
            output_file = lang_output_dir / 'plain-text' / 'merged' / conllu_file.name.replace('.conllu', '.txt')

            texts, mapping = process_conllu_file(conllu_file, output_file, relative_path)

            if mapping:
                file_name = conllu_file.stem
                all_mappings['merged'][file_name] = mapping
                statistics['merged'][file_name] = len(texts)
                statistics['total_files'] += 1
                statistics['total_sentences'] += len(texts)

    # Save mappings
    logger.info("\nSaving mappings...")

    # by_domain mapping
    if all_mappings['by_domain']:
        mapping_file = lang_output_dir / 'mappings' / f'{language.lower()}_by_domain_mappings.json'
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(all_mappings['by_domain'], f, indent=2, ensure_ascii=False)
        logger.info(f"  ✓ Saved: {mapping_file.relative_to(output_base.parent)}")

    # by_file mapping
    if all_mappings['by_file']:
        mapping_file = lang_output_dir / 'mappings' / f'{language.lower()}_by_file_mappings.json'
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(all_mappings['by_file'], f, indent=2, ensure_ascii=False)
        logger.info(f"  ✓ Saved: {mapping_file.relative_to(output_base.parent)}")

    # merged mapping
    if all_mappings['merged']:
        mapping_file = lang_output_dir / 'mappings' / f'{language.lower()}_merged_mappings.json'
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(all_mappings['merged'], f, indent=2, ensure_ascii=False)
        logger.info(f"  ✓ Saved: {mapping_file.relative_to(output_base.parent)}")

    # Save statistics
    stats_file = lang_output_dir / 'mappings' / f'{language.lower()}_extraction_stats.json'
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(statistics, f, indent=2, ensure_ascii=False)
    logger.info(f"  ✓ Saved: {stats_file.relative_to(output_base.parent)}")

    logger.info("\n" + "="*70)
    logger.info(f"{language} - Summary")
    logger.info("="*70)
    logger.info(f"Total files processed: {statistics['total_files']}")
    logger.info(f"Total sentences extracted: {statistics['total_sentences']:,}")
    logger.info("="*70 + "\n")

    return statistics


def main():
    """Main extraction function."""
    # Define paths
    base_dir = Path(__file__).parent.parent
    input_base = base_dir / 'input' / 'converted'
    output_base = base_dir / 'input' / 'converted'

    if not input_base.exists():
        logger.error(f"Input directory not found: {input_base}")
        return

    logger.info("="*70)
    logger.info("PLAIN TEXT EXTRACTION FROM CONLL-U FILES")
    logger.info("="*70)
    logger.info(f"Input base: {input_base}")
    logger.info(f"Output base: {output_base}")
    logger.info("="*70 + "\n")

    # Find all language directories with CoNLL-U files
    languages = []
    for item in sorted(input_base.iterdir()):
        if item.is_dir():
            # Check if it has CoNLL-U files
            has_conllu = (
                list((item / 'by_domain').glob('*.conllu')) if (item / 'by_domain').exists() else [] or
                list((item / 'by_file').rglob('*.conllu')) if (item / 'by_file').exists() else [] or
                list((item / 'merged').glob('*.conllu')) if (item / 'merged').exists() else []
            )
            if has_conllu:
                languages.append(item.name)

    logger.info(f"Found {len(languages)} language(s) with CoNLL-U files: {', '.join(languages)}\n")

    # Process each language
    all_statistics = {}
    for language in languages:
        stats = process_language_directory(language, input_base, output_base)
        all_statistics[language] = stats

    # Save overall summary
    summary_file = output_base / 'extraction_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_statistics, f, indent=2, ensure_ascii=False)

    logger.info("="*70)
    logger.info("EXTRACTION COMPLETE")
    logger.info("="*70)
    logger.info(f"Summary saved to: {summary_file}")

    # Print overall summary
    total_sentences = sum(stats['total_sentences'] for stats in all_statistics.values())
    total_files = sum(stats['total_files'] for stats in all_statistics.values())
    logger.info(f"\nOverall totals:")
    logger.info(f"  Languages: {len(all_statistics)}")
    logger.info(f"  Files: {total_files}")
    logger.info(f"  Sentences: {total_sentences:,}")
    logger.info("="*70)


if __name__ == '__main__':
    main()
