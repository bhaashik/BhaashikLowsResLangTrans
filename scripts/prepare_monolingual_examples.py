#!/usr/bin/env python3
"""
Prepare monolingual examples from PV-BMM-Public-Data for prompt caching.

This script:
1. Loads monolingual sentences from Bhojpuri, Magahi, and Maithili corpora
2. Samples representative sentences
3. Creates example JSON files for use with OpenAI prompt caching
"""

import json
import random
from pathlib import Path
from typing import List, Dict


def load_sentences(file_path: Path, max_sentences: int = 1000) -> List[str]:
    """
    Load sentences from monolingual file.

    Args:
        file_path: Path to text file
        max_sentences: Maximum sentences to load

    Returns:
        List of sentences
    """
    sentences = []

    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_sentences:
                break

            line = line.strip()
            if line and len(line) > 10:  # Skip very short lines
                sentences.append(line)

    return sentences


def sample_diverse_sentences(
    sentences: List[str],
    num_samples: int = 50,
    seed: int = 42
) -> List[str]:
    """
    Sample diverse sentences (short, medium, long).

    Args:
        sentences: List of all sentences
        num_samples: Number to sample
        seed: Random seed

    Returns:
        Sampled sentences
    """
    random.seed(seed)

    # Categorize by length
    short = [s for s in sentences if len(s) < 100]
    medium = [s for s in sentences if 100 <= len(s) < 300]
    long = [s for s in sentences if len(s) >= 300]

    # Sample proportionally
    samples = []
    if short:
        samples.extend(random.sample(short, min(num_samples // 3, len(short))))
    if medium:
        samples.extend(random.sample(medium, min(num_samples // 3, len(medium))))
    if long:
        samples.extend(random.sample(long, min(num_samples // 3, len(long))))

    # Fill up to num_samples if needed
    all_remaining = [s for s in sentences if s not in samples]
    if len(samples) < num_samples and all_remaining:
        samples.extend(random.sample(all_remaining, min(num_samples - len(samples), len(all_remaining))))

    return samples[:num_samples]


def create_example_file(
    language_code: str,
    language_name: str,
    sentences: List[str],
    output_path: Path
):
    """
    Create example JSON file for a language.

    Args:
        language_code: Language code (e.g., 'bho')
        language_name: Language name (e.g., 'Bhojpuri')
        sentences: List of example sentences
        output_path: Output JSON file path
    """
    # Create examples (monolingual, so source = target)
    examples = [
        {
            "text": sentence,
            "language": language_code,
            "language_name": language_name,
            "note": "Monolingual example demonstrating natural language usage"
        }
        for sentence in sentences
    ]

    # Create metadata
    data = {
        "examples": examples,
        "metadata": {
            "language": language_code,
            "language_name": language_name,
            "num_examples": len(examples),
            "source": "PV-BMM-Public-Data",
            "purpose": "OpenAI prompt caching - demonstrate target language patterns",
            "created_date": "2025-12-02",
            "description": f"Representative {language_name} sentences for prompt engineering and caching"
        }
    }

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Created: {output_path}")
    print(f"  Examples: {len(examples)}")
    print(f"  Language: {language_name} ({language_code})")


def main():
    """Main function."""
    print("="*70)
    print("PREPARING MONOLINGUAL EXAMPLES FOR PROMPT CACHING")
    print("="*70)

    # Paths
    base_dir = Path(__file__).parent.parent
    input_dir = base_dir / 'input' / 'PV-BMM-Public-Data'
    output_dir = base_dir / 'universal_translate' / 'data' / 'examples'

    # Language configurations
    languages = [
        {
            'file': 'bhojpuri-all.txt',
            'code': 'bho',
            'name': 'Bhojpuri',
            'num_samples': 50
        },
        {
            'file': 'magahi-all.txt',
            'code': 'mag',
            'name': 'Magahi',
            'num_samples': 50
        },
        {
            'file': 'maithili-all.txt',
            'code': 'mai',
            'name': 'Maithili',
            'num_samples': 50
        }
    ]

    # Process each language
    for lang in languages:
        print(f"\nProcessing {lang['name']}...")

        input_file = input_dir / lang['file']
        if not input_file.exists():
            print(f"  ✗ File not found: {input_file}")
            continue

        # Load sentences
        print(f"  Loading sentences from {input_file.name}...")
        sentences = load_sentences(input_file, max_sentences=10000)
        print(f"  Loaded: {len(sentences)} sentences")

        # Sample diverse sentences
        print(f"  Sampling {lang['num_samples']} diverse sentences...")
        sampled = sample_diverse_sentences(sentences, num_samples=lang['num_samples'])
        print(f"  Sampled: {len(sampled)} sentences")

        # Create example file
        output_file = output_dir / f"hi_to_{lang['code']}_monolingual.json"
        create_example_file(
            language_code=lang['code'],
            language_name=lang['name'],
            sentences=sampled,
            output_path=output_file
        )

    print("\n" + "="*70)
    print("EXAMPLE PREPARATION COMPLETE")
    print("="*70)
    print(f"\nExample files created in: {output_dir}")
    print("\nThese files can be used for:")
    print("1. OpenAI prompt caching (reduces costs significantly)")
    print("2. Demonstrating target language patterns")
    print("3. Providing context for translation quality")


if __name__ == '__main__':
    main()
