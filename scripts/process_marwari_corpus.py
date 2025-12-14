#!/usr/bin/env python3
"""
Process Marwari corpus data from multiple CSV sources.

This script:
1. Loads data from multiple CSV files
2. Cleans HTML tags and noise
3. Filters out navigation menus and structural elements
4. Separates prose from poetry
5. Extracts titles and author information
6. Creates metadata for provenance tracking
7. Generates structured monolingual corpus
"""

import csv
import json
import re
import os
import sys
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import html

# Increase CSV field size limit to handle large fields
csv.field_size_limit(sys.maxsize)


class MarwariCorpusProcessor:
    """Process and clean Marwari corpus from multiple sources."""

    # Navigation menu keywords to filter out (common across pages)
    NAVIGATION_KEYWORDS = {
        'आपाणो राजस्थान', 'पेहलो पानो', 'आपरी राय', 'सहायता', 'माणो पतो',
        'समाचार', 'धरती धोरा री', 'धरती मगरा री', 'धरती चंबल री',
        'धरती मीरा री', 'धरती वीरा री', 'राजस्थान रा जिला रो नक्शो',
        'सेटेलाइट', 'लोकदेवता', 'महान विभूतिया', 'साहित्यकार',
        'विद्वाना रा विचार', 'राजनैतिक परिपेक्ष्य', 'विकास योजनायां',
        'जिला री जाणकार', 'खान-पान', 'शिक्षा व संस्थान', 'घुमवा री जगहा'
    }

    # District names to filter
    DISTRICTS = {
        'अजमेर', 'अलवर', 'उदयपुर', 'करौली', 'कोटा', 'गंगानगर', 'चितौड़गढ़',
        'चूरु', 'जयपुर', 'जालौर', 'जैसलमेर', 'जोधपुर', 'झालावाड़', 'झुन्झुनूं',
        'टोंक', 'डूँग़रपुर', 'दौसा', 'धौलपुर', 'नागौर', 'पाली', 'प्रतापगढ़',
        'बाड़मेर', 'बाँसवाड़ा', 'बीकानेर', 'बूँदी', 'बारां', 'भरतपुर',
        'सवाई माधोपुर', 'राजसमन्द', 'भीलवाड़ा', 'सिरोही', 'सीकर', 'ह्नुमानगढ़'
    }

    def __init__(self, source_dir: str, output_dir: str):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.metadata_dir = self.output_dir / 'metadata'
        self.prose_dir = self.output_dir / 'prose'
        self.poetry_dir = self.output_dir / 'poetry'

        # Create output directories
        for dir_path in [self.metadata_dir, self.prose_dir, self.poetry_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Storage for processed data
        self.prose_entries = []
        self.poetry_entries = []
        self.metadata = []

    def clean_html(self, text: str) -> str:
        """Remove HTML tags and decode HTML entities."""
        if not text:
            return ""

        # Decode HTML entities
        text = html.unescape(text)

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Remove common HTML artifacts
        text = re.sub(r'&nbsp;', ' ', text)
        text = re.sub(r'&[a-z]+;', '', text)

        return text.strip()

    def is_navigation_item(self, text: str) -> bool:
        """Check if text is a navigation menu item."""
        if not text or len(text.strip()) < 3:
            return True

        # Check against navigation keywords
        if text.strip() in self.NAVIGATION_KEYWORDS:
            return True

        # Check against district names
        if text.strip() in self.DISTRICTS:
            return True

        # Single word or very short items are likely navigation
        words = text.strip().split()
        if len(words) <= 2 and len(text) < 30:
            return True

        # Items that are just punctuation or special characters
        if re.match(r'^[^\w\s]+$', text.strip()):
            return True

        # URLs and email addresses
        if re.search(r'(https?://|www\.|@|\.(com|org|in))', text):
            return True

        # Phone numbers
        if re.search(r'\d{10,}|\d{3,5}-\d{4,8}', text):
            return True

        return False

    def is_poetry(self, text: str) -> bool:
        """Heuristically determine if text is poetry."""
        if not text or len(text.strip()) < 10:
            return False

        lines = text.strip().split('\n')

        # Poetry typically has multiple short lines
        if len(lines) > 2:
            avg_line_length = sum(len(line.strip()) for line in lines) / len(lines)
            if avg_line_length < 50:  # Short lines suggest poetry
                return True

        # Check for common poetry markers
        poetry_markers = ['॥', '।।', 'छंद', 'दोहा', 'कविता', 'पद्य']
        if any(marker in text for marker in poetry_markers):
            return True

        # Check for meter/rhythm (repeating patterns)
        # This is a simple heuristic - could be improved
        if re.search(r'(\w+)\s+(\w+)\s+(\w+)\s+।', text):
            return True

        return False

    def extract_title_author(self, text: str, url: str = "") -> Tuple[Optional[str], Optional[str]]:
        """Extract title and author from text if present."""
        title = None
        author = None

        # Look for title patterns
        title_match = re.search(r'^([^\n।]+?)\s*[\n।]', text)
        if title_match and len(title_match.group(1)) < 100:
            potential_title = title_match.group(1).strip()
            # Validate it looks like a title (capitalized, not too long)
            if not self.is_navigation_item(potential_title):
                title = potential_title

        # Look for author patterns
        author_patterns = [
            r'लेखक[:\s]+([^\n।]+)',
            r'रचनाकार[:\s]+([^\n।]+)',
            r'कवि[:\s]+([^\n।]+)',
            r'द्वारा[:\s]+([^\n।]+)',
            r'-\s*([^\n]{5,40})\s*$',  # Author at end with dash
        ]

        for pattern in author_patterns:
            author_match = re.search(pattern, text)
            if author_match:
                potential_author = author_match.group(1).strip()
                if len(potential_author) < 50 and not self.is_navigation_item(potential_author):
                    author = potential_author
                    break

        return title, author

    def clean_content(self, text: str) -> str:
        """Remove titles, authors, and other metadata from content."""
        # Remove common metadata patterns
        text = re.sub(r'^[^\n।]+?[:\-]\s*', '', text)  # Remove leading titles
        text = re.sub(r'लेखक[:\s]+[^\n।]+[\n।]', '', text)
        text = re.sub(r'रचनाकार[:\s]+[^\n।]+[\n।]', '', text)

        # Clean up whitespace
        text = re.sub(r'\n\s*\n', '\n', text)
        text = re.sub(r'[ \t]+', ' ', text)

        return text.strip()

    def process_csv_file(self, csv_path: Path, source_name: str):
        """Process a single CSV file."""
        print(f"Processing {csv_path.name}...")

        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                # Try to detect delimiter
                sample = f.read(1024)
                f.seek(0)

                # Use csv.Sniffer to detect format
                try:
                    sniffer = csv.Sniffer()
                    delimiter = sniffer.sniff(sample).delimiter
                except:
                    delimiter = ','

                reader = csv.DictReader(f, delimiter=delimiter)

                for row_num, row in enumerate(reader, 1):
                    # Extract text from different possible column names
                    text = row.get('text', row.get('marwari_text', ''))
                    url = row.get('url', '')
                    title_col = row.get('title', '')

                    if not text:
                        continue

                    # Clean HTML
                    text = self.clean_html(text)

                    # Split by newlines and process each segment
                    segments = [s.strip() for s in text.split('\n') if s.strip()]

                    for seg in segments:
                        # Skip navigation items
                        if self.is_navigation_item(seg):
                            continue

                        # Skip very short segments
                        if len(seg) < 20:
                            continue

                        # Extract metadata
                        title, author = self.extract_title_author(seg, url)

                        # Clean content
                        content = self.clean_content(seg)

                        if len(content) < 20:
                            continue

                        # Create entry
                        entry = {
                            'text': content,
                            'source': source_name,
                            'source_file': csv_path.name,
                            'url': url,
                            'title': title or title_col or None,
                            'author': author,
                            'row_number': row_num
                        }

                        # Classify as prose or poetry
                        if self.is_poetry(content):
                            self.poetry_entries.append(entry)
                        else:
                            self.prose_entries.append(entry)

                        # Add to metadata
                        self.metadata.append(entry)

        except Exception as e:
            print(f"Error processing {csv_path.name}: {e}")

    def save_corpus(self):
        """Save processed corpus to files."""
        print(f"\nSaving corpus...")
        print(f"Prose entries: {len(self.prose_entries)}")
        print(f"Poetry entries: {len(self.poetry_entries)}")

        # Save prose
        prose_file = self.prose_dir / 'marwari_prose.txt'
        with open(prose_file, 'w', encoding='utf-8') as f:
            for entry in self.prose_entries:
                if entry.get('title'):
                    f.write(f"# {entry['title']}\n")
                if entry.get('author'):
                    f.write(f"## {entry['author']}\n")
                f.write(f"{entry['text']}\n\n")

        print(f"Saved prose to: {prose_file}")

        # Save poetry
        poetry_file = self.poetry_dir / 'marwari_poetry.txt'
        with open(poetry_file, 'w', encoding='utf-8') as f:
            for entry in self.poetry_entries:
                if entry.get('title'):
                    f.write(f"# {entry['title']}\n")
                if entry.get('author'):
                    f.write(f"## {entry['author']}\n")
                f.write(f"{entry['text']}\n\n")

        print(f"Saved poetry to: {poetry_file}")

        # Save metadata
        metadata_file = self.metadata_dir / 'marwari_metadata.jsonl'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            for entry in self.metadata:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

        print(f"Saved metadata to: {metadata_file}")

        # Save statistics
        stats = {
            'total_entries': len(self.metadata),
            'prose_entries': len(self.prose_entries),
            'poetry_entries': len(self.poetry_entries),
            'sources': list(set(e['source'] for e in self.metadata)),
            'entries_with_titles': sum(1 for e in self.metadata if e.get('title')),
            'entries_with_authors': sum(1 for e in self.metadata if e.get('author'))
        }

        stats_file = self.metadata_dir / 'marwari_stats.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        print(f"Saved statistics to: {stats_file}")
        print(f"\nStatistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    def process_all(self):
        """Process all CSV files in the source directory."""
        csv_files = list(self.source_dir.glob('*.csv'))

        if not csv_files:
            print(f"No CSV files found in {self.source_dir}")
            return

        print(f"Found {len(csv_files)} CSV files to process\n")

        for csv_file in csv_files:
            source_name = csv_file.stem
            self.process_csv_file(csv_file, source_name)

        self.save_corpus()


def main():
    """Main entry point."""
    # Set up paths
    base_dir = Path(__file__).parent.parent
    source_dir = base_dir / 'misc' / 'MarwariCorpus-main' / 'marwariext'
    output_dir = base_dir / 'universal_translate' / 'data' / 'monolingual' / 'marwari'

    print("="*60)
    print("Marwari Corpus Processor")
    print("="*60)
    print(f"Source directory: {source_dir}")
    print(f"Output directory: {output_dir}")
    print("="*60 + "\n")

    # Create processor and run
    processor = MarwariCorpusProcessor(source_dir, output_dir)
    processor.process_all()

    print("\n" + "="*60)
    print("Processing complete!")
    print("="*60)


if __name__ == '__main__':
    main()
