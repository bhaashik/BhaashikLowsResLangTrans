#!/usr/bin/env python3
"""
Extract monolingual examples from VarDial 2018 dataset for Awadhi and Braj.

VarDial 2018 contains 15,000 sentences each in:
- Awadhi (awa)
- Braj (bra)
- Bhojpuri (bho)
- Magahi (mag)
- Hindi (hin)

Source: https://github.com/kmi-linguistics/vardial2018
"""

import json
import random
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm


# Language configurations
LANGUAGES = {
    'awa': {
        'name': 'Awadhi',
        'vardial_file': 'vardial2018/data/AWA.txt',  # Adjust based on actual structure
        'num_examples': 100,
        'output_file': 'universal_translate/data/examples/hi_to_awa_monolingual.json'
    },
    'bra': {
        'name': 'Braj',
        'vardial_file': 'vardial2018/data/BRA.txt',
        'num_examples': 100,
        'output_file': 'universal_translate/data/examples/hi_to_bra_monolingual.json'
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
        if not sent or len(sent) < min_length or len(sent) > max_length:
            continue

        # Skip sentences that are mostly numbers or punctuation
        alpha_ratio = sum(c.isalpha() for c in sent) / len(sent)
        if alpha_ratio < 0.5:
            continue

        # Skip if too many repeating characters
        if any(sent.count(c) > len(sent) * 0.3 for c in set(sent) if c.isalpha()):
            continue

        filtered.append(sent)

    return filtered


def extract_from_vardial(lang_code: str, config: Dict, vardial_dir: Path = Path("vardial2018")) -> List[Dict]:
    """
    Extract examples from VarDial 2018 dataset.

    Args:
        lang_code: Language code (e.g., 'awa', 'bra')
        config: Language configuration
        vardial_dir: Path to VarDial dataset directory

    Returns:
        List of example dictionaries
    """
    print(f"Loading VarDial dataset for {config['name']}...")

    # Check if VarDial directory exists
    if not vardial_dir.exists():
        print(f"Error: VarDial directory not found: {vardial_dir}")
        print(f"Please clone it first:")
        print(f"  git clone https://github.com/kmi-linguistics/vardial2018.git")
        return []

    # Try different possible file paths
    possible_paths = [
        vardial_dir / "data" / f"{lang_code.upper()}.txt",
        vardial_dir / "data" / f"{lang_code}.txt",
        vardial_dir / f"{lang_code.upper()}.txt",
        vardial_dir / f"{lang_code}.txt",
        vardial_dir / "train" / f"{lang_code.upper()}.txt",
        vardial_dir / "train" / f"{lang_code}.txt",
    ]

    data_file = None
    for path in possible_paths:
        if path.exists():
            data_file = path
            break

    if not data_file:
        print(f"Error: Could not find data file for {config['name']}")
        print(f"Checked paths:")
        for path in possible_paths:
            print(f"  - {path}")
        return []

    print(f"Reading from: {data_file}")

    # Read sentences
    sentences = []
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    # VarDial format might have labels, extract just the text
                    # Format could be: "label\tsentence" or just "sentence"
                    if '\t' in line:
                        parts = line.split('\t')
                        sentence = parts[-1]  # Take last part as sentence
                    else:
                        sentence = line
                    sentences.append(sentence)
    except Exception as e:
        print(f"Error reading file: {e}")
        return []

    print(f"Read {len(sentences)} sentences")

    # Filter for quality
    print(f"Filtering sentences...")
    filtered = filter_quality_sentences(sentences)
    print(f"After filtering: {len(filtered)} sentences")

    # Randomly sample
    if len(filtered) > config['num_examples']:
        filtered = random.sample(filtered, config['num_examples'])
    elif len(filtered) < config['num_examples']:
        print(f"Warning: Only {len(filtered)} quality sentences available (wanted {config['num_examples']})")

    # Create example objects
    examples = [
        {
            "text": sent,
            "language": lang_code,
            "language_name": config['name'],
            "source": "VarDial 2018",
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
        "source": "VarDial 2018 Indo-Aryan Language Identification Dataset",
        "dataset": "https://github.com/kmi-linguistics/vardial2018",
        "language": lang_code,
        "language_name": lang_name,
        "note": "Monolingual examples for use in translation prompts with caching",
        "num_examples": len(examples),
        "domain": "Literature (primarily)",
        "examples": examples
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✓ Saved {len(examples)} examples to {output_file}")


def main():
    """Main extraction workflow."""
    random.seed(42)  # For reproducibility

    print("=" * 60)
    print("VarDial 2018 Example Extractor")
    print("=" * 60)
    print()

    # Check if VarDial directory exists
    vardial_dir = Path("vardial2018")
    if not vardial_dir.exists():
        print("VarDial 2018 dataset not found!")
        print()
        print("To download:")
        print("  git clone https://github.com/kmi-linguistics/vardial2018.git")
        print()
        return

    # Extract examples
    for lang_code, config in LANGUAGES.items():
        print(f"\nProcessing {config['name']} ({lang_code})...")
        print("-" * 60)

        examples = extract_from_vardial(lang_code, config, vardial_dir)

        if examples:
            save_examples(examples, config['output_file'], lang_code, config['name'])
        else:
            print(f"✗ No examples extracted for {config['name']}")

    print("\n" + "=" * 60)
    print("Extraction complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Verify the examples:")
    print("   cat universal_translate/data/examples/hi_to_awa_monolingual.json | jq '.num_examples'")
    print("2. Test translation:")
    print("   python examples/translate_with_openai_caching.py")


if __name__ == "__main__":
    main()
