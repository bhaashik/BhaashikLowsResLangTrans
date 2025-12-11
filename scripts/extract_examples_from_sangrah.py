#!/usr/bin/env python3
"""
Extract monolingual examples from AI4Bharat's Sangrah dataset.

The Sangrah dataset contains monolingual data for various Indic languages including:
- Nepali (nep)
- Maithili (mai)
- Marwari (mwr)
- And others

This script extracts high-quality examples for use in translation prompts with caching.
"""

import json
import random
from pathlib import Path
from typing import List, Dict
from datasets import load_dataset
from tqdm import tqdm

# Language configurations
LANGUAGES = {
    'nep': {
        'name': 'Nepali',
        'sangrah_key': 'nep',  # Check actual dataset for correct key
        'num_examples': 100,
        'output_file': 'universal_translate/data/examples/hi_to_nep_monolingual.json'
    },
    'mai': {
        'name': 'Maithili',
        'sangrah_key': 'mai',
        'num_examples': 100,
        'output_file': 'universal_translate/data/examples/hi_to_mai_monolingual.json'
    },
    'mwr': {
        'name': 'Marwari',
        'sangrah_key': 'mwr',
        'num_examples': 100,
        'output_file': 'universal_translate/data/examples/hi_to_mwr_monolingual.json'
    }
}

# For languages not in Sangrah, we can use other sources
LOCAL_CORPUS_LANGUAGES = {
    'awa': {
        'name': 'Awadhi',
        'num_examples': 100,
        'output_file': 'universal_translate/data/examples/hi_to_awa_monolingual.json',
        'corpus_path': 'input/awadhi_corpus/'  # If you have local data
    },
    'bra': {
        'name': 'Braj',
        'num_examples': 100,
        'output_file': 'universal_translate/data/examples/hi_to_bra_monolingual.json',
        'corpus_path': 'input/braj_corpus/'
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
        if not sent or len(sent) < min_length or len(sent) > max_length:
            continue

        # Skip sentences that are mostly numbers or punctuation
        alpha_ratio = sum(c.isalpha() for c in sent) / len(sent)
        if alpha_ratio < 0.5:
            continue

        # Skip if too many repeating characters
        if any(sent.count(c) > len(sent) * 0.3 for c in set(sent)):
            continue

        filtered.append(sent.strip())

    return filtered


def extract_from_sangrah(lang_code: str, config: Dict) -> List[Dict]:
    """
    Extract examples from Sangrah dataset.

    Args:
        lang_code: Language code (e.g., 'nep', 'mai', 'mwr')
        config: Language configuration

    Returns:
        List of example dictionaries
    """
    print(f"Loading Sangrah dataset for {config['name']}...")

    try:
        # Load dataset - adjust subset name based on actual Sangrah structure
        dataset = load_dataset(
            "ai4bharat/sangrah",
            config['sangrah_key'],
            split="train",
            streaming=True  # Use streaming for large datasets
        )

        sentences = []
        print(f"Extracting sentences for {config['name']}...")

        # Collect sentences
        for i, example in enumerate(tqdm(dataset, desc="Processing")):
            # Adjust field name based on actual dataset structure
            # Common fields: 'text', 'sentence', 'content'
            text = example.get('text') or example.get('sentence') or example.get('content')

            if text and isinstance(text, str):
                sentences.append(text)

            # Stop after collecting enough
            if len(sentences) >= config['num_examples'] * 5:  # Get 5x for filtering
                break

        # Filter for quality
        print(f"Filtering {len(sentences)} sentences...")
        filtered = filter_quality_sentences(sentences)

        # Randomly sample
        if len(filtered) > config['num_examples']:
            filtered = random.sample(filtered, config['num_examples'])

        # Create example objects
        examples = [
            {
                "text": sent,
                "language": lang_code,
                "language_name": config['name'],
                "source": "ai4bharat/sangrah",
                "note": "Monolingual example for prompt caching"
            }
            for sent in filtered
        ]

        return examples

    except Exception as e:
        print(f"Error loading Sangrah dataset for {lang_code}: {e}")
        print(f"You may need to check the dataset structure or authentication.")
        return []


def extract_from_local_corpus(lang_code: str, config: Dict) -> List[Dict]:
    """
    Extract examples from local corpus files.

    Args:
        lang_code: Language code
        config: Language configuration

    Returns:
        List of example dictionaries
    """
    corpus_path = Path(config['corpus_path'])

    if not corpus_path.exists():
        print(f"Warning: Corpus path not found: {corpus_path}")
        print(f"Please add {config['name']} corpus data to {corpus_path}")
        return []

    sentences = []

    # Read from text files
    for txt_file in corpus_path.glob('**/*.txt'):
        with open(txt_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    sentences.append(line)

    # Filter and sample
    filtered = filter_quality_sentences(sentences)
    if len(filtered) > config['num_examples']:
        filtered = random.sample(filtered, config['num_examples'])

    examples = [
        {
            "text": sent,
            "language": lang_code,
            "language_name": config['name'],
            "source": "local_corpus",
            "note": "Monolingual example for prompt caching"
        }
        for sent in filtered
    ]

    return examples


def save_examples(examples: List[Dict], output_file: str, lang_code: str, lang_name: str):
    """
    Save examples to JSON file.

    Args:
        examples: List of example dictionaries
        output_file: Output file path
        lang_code: Language code
        lang_name: Language name
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "source": "ai4bharat/sangrah or local corpus",
        "dataset": "ai4bharat/sangrah",
        "language": lang_code,
        "language_name": lang_name,
        "note": "Monolingual examples for use in translation prompts with caching",
        "num_examples": len(examples),
        "examples": examples
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(examples)} examples to {output_file}")


def main():
    """Main extraction workflow."""
    random.seed(42)  # For reproducibility

    print("=" * 60)
    print("Sangrah Dataset Example Extractor")
    print("=" * 60)
    print()

    # Extract from Sangrah dataset
    for lang_code, config in LANGUAGES.items():
        print(f"\nProcessing {config['name']} ({lang_code})...")
        print("-" * 60)

        examples = extract_from_sangrah(lang_code, config)

        if examples:
            save_examples(examples, config['output_file'], lang_code, config['name'])
        else:
            print(f"No examples extracted for {config['name']}")

    # Extract from local corpus
    for lang_code, config in LOCAL_CORPUS_LANGUAGES.items():
        print(f"\nProcessing {config['name']} ({lang_code}) from local corpus...")
        print("-" * 60)

        examples = extract_from_local_corpus(lang_code, config)

        if examples:
            save_examples(examples, config['output_file'], lang_code, config['name'])
        else:
            print(f"No examples extracted for {config['name']}")

    print("\n" + "=" * 60)
    print("Extraction complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
