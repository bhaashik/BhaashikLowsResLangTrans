#!/usr/bin/env python3
"""
Extract 100 high-quality Marwari examples for prompt caching.

Filters out navigation items and selects quality prose sentences.
"""

import json
import random
import re
from pathlib import Path
from typing import List, Dict


def load_metadata(metadata_file: Path) -> List[Dict]:
    """Load metadata from JSONL file."""
    entries = []
    with open(metadata_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                entries.append(entry)
            except json.JSONDecodeError:
                continue
    return entries


def is_quality_prose(text: str) -> bool:
    """Check if text is quality prose (not navigation, not junk)."""
    if not text or len(text.strip()) < 40:
        return False

    # Must have at least 5 words
    words = text.strip().split()
    if len(words) < 5:
        return False

    # Check alpha ratio
    alpha_chars = sum(c.isalpha() or c.isspace() for c in text)
    if alpha_chars / len(text) < 0.6:
        return False

    # Reject if too many numbers
    digit_ratio = sum(c.isdigit() for c in text) / len(text)
    if digit_ratio > 0.2:
        return False

    # Reject if has URLs, emails, phone numbers
    if re.search(r'(https?://|www\.|@|\d{10,})', text):
        return False

    # Reject if has too many English words (more than 30%)
    english_word_count = sum(1 for word in words if re.match(r'^[a-zA-Z]+$', word))
    if english_word_count / len(words) > 0.3:
        return False

    # Reject if has HTML artifacts
    if re.search(r'(<[^>]+>|&[a-z]+;)', text):
        return False

    # Must have proper Devanagari content
    devanagari_chars = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    if devanagari_chars < 20:
        return False

    # Reject very repetitive text
    if len(set(text.split())) / len(words) < 0.5:
        return False

    return True


def extract_examples(metadata_file: Path, num_examples: int = 100) -> List[Dict]:
    """Extract high-quality examples from metadata."""
    print(f"Loading metadata from {metadata_file}...")
    all_entries = load_metadata(metadata_file)
    print(f"Loaded {len(all_entries)} total entries")

    # Filter for quality prose
    print("Filtering for quality prose...")
    quality_entries = []
    for entry in all_entries:
        text = entry.get('text', '')

        # Skip if marked as poetry (we want prose for translation examples)
        # Skip short entries
        if len(text) < 40:
            continue

        if is_quality_prose(text):
            quality_entries.append(entry)

    print(f"Found {len(quality_entries)} quality prose entries")

    # Remove duplicates by text
    seen_texts = set()
    unique_entries = []
    for entry in quality_entries:
        text = entry['text'].strip()
        if text not in seen_texts:
            seen_texts.add(text)
            unique_entries.append(entry)

    print(f"After deduplication: {len(unique_entries)} unique entries")

    # Sample randomly
    if len(unique_entries) <= num_examples:
        selected = unique_entries
    else:
        # Stratify by source to get diverse examples
        by_source = {}
        for entry in unique_entries:
            source = entry.get('source', 'unknown')
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(entry)

        # Sample proportionally from each source
        selected = []
        per_source = num_examples // len(by_source)
        remainder = num_examples % len(by_source)

        for source, entries in sorted(by_source.items()):
            sample_size = per_source + (1 if remainder > 0 else 0)
            if remainder > 0:
                remainder -= 1

            sample = random.sample(entries, min(sample_size, len(entries)))
            selected.extend(sample)

        # If we still need more, sample from remaining
        if len(selected) < num_examples:
            remaining = [e for e in unique_entries if e not in selected]
            additional = random.sample(remaining, min(num_examples - len(selected), len(remaining)))
            selected.extend(additional)

        selected = selected[:num_examples]

    print(f"Selected {len(selected)} examples")
    return selected


def save_examples(examples: List[Dict], output_file: Path):
    """Save examples to JSON file."""
    # Create example objects
    example_list = []
    for entry in examples:
        example_list.append({
            'text': entry['text'],
            'language': 'mwr',
            'language_name': 'Marwari',
            'source': entry.get('source', 'Unknown'),
            'note': 'Monolingual example for prompt caching'
        })

    # Create output object
    output = {
        'source': 'Marwari Corpus (aapanorajasthan.org and other sources)',
        'repository': 'misc/MarwariCorpus-main',
        'language': 'mwr',
        'language_name': 'Marwari',
        'num_examples': len(example_list),
        'domain': 'Mixed (literature, news, history, culture)',
        'examples': example_list
    }

    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(example_list)} examples to {output_file}")


def main():
    """Main entry point."""
    # Set random seed for reproducibility
    random.seed(42)

    # Paths
    base_dir = Path(__file__).parent.parent
    metadata_file = base_dir / 'universal_translate' / 'data' / 'monolingual' / 'marwari' / 'metadata' / 'marwari_metadata.jsonl'
    output_file = base_dir / 'universal_translate' / 'data' / 'examples' / 'hi_to_mwr_monolingual.json'

    print("="*60)
    print("Marwari Example Extractor")
    print("="*60)
    print(f"Input: {metadata_file}")
    print(f"Output: {output_file}")
    print("="*60 + "\n")

    # Extract examples
    examples = extract_examples(metadata_file, num_examples=100)

    # Save examples
    output_file.parent.mkdir(parents=True, exist_ok=True)
    save_examples(examples, output_file)

    print("\n" + "="*60)
    print("Extraction complete!")
    print("="*60)


if __name__ == '__main__':
    main()
