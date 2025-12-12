#!/usr/bin/env python3
"""
Extract monolingual examples from KMI linguistics corpora.

Processes:
- Braj (from CoNLL-U files)
- Bodo (from text files)
- Awadhi (from text files)

Outputs JSON files with 100 examples each for use in translation prompts.
"""

import json
import random
import re
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm


# Language configurations
LANGUAGES = {
    'awa': {
        'name': 'Awadhi',
        'source_dir': 'misc/awadhi/source',
        'pattern': 'awadhi-corpus-*.txt',
        'num_examples': 100,
        'output_file': 'universal_translate/data/examples/hi_to_awa_monolingual.json',
        'format': 'text'
    },
    'bra': {
        'name': 'Braj',
        'source_files': [
            'misc/braj/data/auto-pos/braj_ud.conll',
            'misc/braj/data/bis_data/train_file.conll',
            'misc/braj/data/ud_data/train_file_ud.conll'
        ],
        'num_examples': 100,
        'output_file': 'universal_translate/data/examples/hi_to_bra_monolingual.json',
        'format': 'conllu'
    },
    'brx': {  # Bodo ISO 639-3 code
        'name': 'Bodo',
        'source_dir': 'misc/bodo/corpus',
        'pattern': 'bodo-corpus-*.txt',
        'num_examples': 100,
        'output_file': 'universal_translate/data/examples/hi_to_brx_monolingual.json',
        'format': 'text'
    }
}


def filter_quality_sentences(sentences: List[str], min_length: int = 20, max_length: int = 200) -> List[str]:
    """
    Filter sentences for quality.

    Args:
        sentences: List of sentences
        min_length: Minimum character length
        max_length: Maximum character length

    Returns:
        Filtered sentences
    """
    filtered = []
    for sent in sentences:
        # Basic quality checks
        sent = sent.strip()

        # Remove BOM and other invisible characters
        sent = sent.replace('\ufeff', '').replace('\u200b', '')

        if not sent or len(sent) < min_length or len(sent) > max_length:
            continue

        # Skip sentences that are mostly numbers or punctuation
        alpha_ratio = sum(c.isalpha() for c in sent) / len(sent) if len(sent) > 0 else 0
        if alpha_ratio < 0.5:
            continue

        # Skip if too many repeating characters
        if any(sent.count(c) > len(sent) * 0.3 for c in set(sent) if c.isalpha()):
            continue

        # Skip very short sentences
        words = sent.split()
        if len(words) < 3:
            continue

        filtered.append(sent)

    return filtered


def extract_from_text_files(source_dir: Path, pattern: str) -> List[str]:
    """
    Extract sentences from plain text files.

    Args:
        source_dir: Directory containing text files
        pattern: Glob pattern for files

    Returns:
        List of sentences
    """
    sentences = []
    files = sorted(source_dir.glob(pattern))

    print(f"  Found {len(files)} files in {source_dir}")

    for file_path in tqdm(files, desc="  Reading files"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

                # Split by common sentence delimiters
                # Handle both Devanagari and Latin punctuation
                sents = re.split(r'[।\.\!\?]+\s*', content)
                sentences.extend([s.strip() for s in sents if s.strip()])

        except Exception as e:
            print(f"  Warning: Error reading {file_path}: {e}")
            continue

    return sentences


def extract_from_conllu(file_paths: List[str]) -> List[str]:
    """
    Extract sentences from CoNLL-U format files.

    Args:
        file_paths: List of CoNLL-U file paths

    Returns:
        List of sentences
    """
    sentences = []

    for file_path in file_paths:
        path = Path(file_path)
        if not path.exists():
            print(f"  Warning: File not found: {file_path}")
            continue

        print(f"  Reading {path.name}")

        try:
            current_sentence = []

            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()

                    # Remove BOM
                    line = line.replace('\ufeff', '')

                    # Empty line marks end of sentence
                    if not line:
                        if current_sentence:
                            # Join tokens to form sentence
                            sentence = ' '.join(current_sentence)
                            sentences.append(sentence)
                            current_sentence = []
                        continue

                    # Skip comments
                    if line.startswith('#'):
                        continue

                    # Try standard CoNLL-U format (10 columns)
                    parts = line.split('\t')
                    if len(parts) >= 10:
                        token_id = parts[0]
                        form = parts[1]

                        # Skip multi-word tokens (e.g., "1-2")
                        if '-' in token_id or '.' in token_id:
                            continue

                        current_sentence.append(form)

                    # Try simplified format (2 columns: WORD POS)
                    elif len(parts) == 2:
                        word = parts[0].strip()
                        pos = parts[1].strip()

                        # Skip punctuation-only tokens
                        if word and not word in ['।', '.', ',', '!', '?']:
                            current_sentence.append(word)
                        elif word in ['।', '.']:  # Sentence-ending punctuation
                            if current_sentence:
                                sentence = ' '.join(current_sentence)
                                sentences.append(sentence)
                                current_sentence = []

                    # Try space-separated format (WORD POS)
                    elif ' ' in line and not '\t' in line:
                        parts = line.split()
                        if len(parts) >= 1:
                            word = parts[0].strip()
                            # Skip punctuation-only tokens
                            if word and not word in ['।', '.', ',', '!', '?']:
                                current_sentence.append(word)
                            elif word in ['।', '.']:  # Sentence-ending punctuation
                                if current_sentence:
                                    sentence = ' '.join(current_sentence)
                                    sentences.append(sentence)
                                    current_sentence = []

            # Add last sentence if exists
            if current_sentence:
                sentence = ' '.join(current_sentence)
                sentences.append(sentence)

        except Exception as e:
            print(f"  Error reading {file_path}: {e}")
            continue

    return sentences


def process_language(lang_code: str, config: Dict) -> List[Dict]:
    """
    Process a language corpus and extract examples.

    Args:
        lang_code: Language code
        config: Language configuration

    Returns:
        List of example dictionaries
    """
    print(f"\nProcessing {config['name']} ({lang_code})...")
    print("-" * 60)

    # Extract sentences based on format
    if config['format'] == 'text':
        source_dir = Path(config['source_dir'])
        if not source_dir.exists():
            print(f"  Error: Source directory not found: {source_dir}")
            return []

        sentences = extract_from_text_files(source_dir, config['pattern'])

    elif config['format'] == 'conllu':
        sentences = extract_from_conllu(config['source_files'])

    else:
        print(f"  Error: Unknown format: {config['format']}")
        return []

    print(f"  Extracted {len(sentences)} sentences")

    # Filter for quality
    print(f"  Filtering sentences...")
    filtered = filter_quality_sentences(sentences)
    print(f"  After filtering: {len(filtered)} sentences")

    # Remove duplicates
    filtered = list(set(filtered))
    print(f"  After deduplication: {len(filtered)} sentences")

    # Randomly sample
    num_examples = min(config['num_examples'], len(filtered))
    if len(filtered) > num_examples:
        sampled = random.sample(filtered, num_examples)
    else:
        sampled = filtered
        print(f"  Warning: Only {len(filtered)} quality sentences available (wanted {config['num_examples']})")

    # Create example objects
    examples = [
        {
            "text": sent,
            "language": lang_code,
            "language_name": config['name'],
            "source": "KMI Linguistics Corpus",
            "note": "Monolingual example for prompt caching"
        }
        for sent in sampled
    ]

    return examples


def save_examples(examples: List[Dict], output_file: str, lang_code: str, lang_name: str, source: str):
    """
    Save examples to JSON file.

    Args:
        examples: List of example dictionaries
        output_file: Output file path
        lang_code: Language code
        lang_name: Language name
        source: Data source description
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "source": source,
        "repository": "https://github.com/kmi-linguistics/",
        "language": lang_code,
        "language_name": lang_name,
        "note": "Monolingual examples for use in translation prompts with caching",
        "num_examples": len(examples),
        "domain": "Mixed (literature, news, general)",
        "examples": examples
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"  ✓ Saved {len(examples)} examples to {output_file}")


def main():
    """Main extraction workflow."""
    random.seed(42)  # For reproducibility

    print("=" * 60)
    print("KMI Linguistics Corpus Example Extractor")
    print("=" * 60)
    print()

    # Process each language
    for lang_code, config in LANGUAGES.items():
        examples = process_language(lang_code, config)

        if examples:
            save_examples(
                examples,
                config['output_file'],
                lang_code,
                config['name'],
                "KMI Linguistics Corpus"
            )
        else:
            print(f"  ✗ No examples extracted for {config['name']}")

    print("\n" + "=" * 60)
    print("Extraction complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Verify the examples:")
    print("   cat universal_translate/data/examples/hi_to_awa_monolingual.json | jq '.num_examples'")
    print("   cat universal_translate/data/examples/hi_to_bra_monolingual.json | jq '.num_examples'")
    print("   cat universal_translate/data/examples/hi_to_brx_monolingual.json | jq '.num_examples'")
    print()
    print("2. Test translation:")
    print("   python scripts/translate_cli.py --provider openai --source-lang hi --target-lang awa --text 'मैं घर जा रहा हूं।'")


if __name__ == "__main__":
    main()
