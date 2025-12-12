#!/usr/bin/env python3
"""
Kavita Kosh Crawler - Simple BeautifulSoup-based approach
Better for research, experimentation, and Jupyter notebooks
"""

import requests
from bs4 import BeautifulSoup
from pathlib import Path
import json
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from urllib.parse import quote, unquote, urlparse, parse_qs
from collections import defaultdict


@dataclass
class Poem:
    """Data class for poem metadata and content"""
    title: str
    author: str
    text: str
    language: str
    url: str
    categories: List[str] = None
    download_time: str = None
    content_length: int = 0
    
    def __post_init__(self):
        if self.categories is None:
            self.categories = []
        self.content_length = len(self.text)


class KavitakoshCrawler:
    """
    Simple crawler for Kavita Kosh
    Designed for interactive use and research
    """
    
    def __init__(self, output_dir: str = 'kavitakosh_data', delay: float = 1.0):
        """
        Initialize crawler
        
        Args:
            output_dir: Directory to save poems
            delay: Delay between requests in seconds (respectful crawling)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.delay = delay
        self.session = self._create_session()
        self.poems_data: List[Poem] = []
        self.stats = defaultdict(int)
        
    def _create_session(self) -> requests.Session:
        """Create a requests session with headers"""
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; NLP-Research-Bot/1.0)',
        })
        return session
    
    def _fetch_page(self, url: str) -> Optional[BeautifulSoup]:
        """
        Fetch a page and return BeautifulSoup object
        
        Args:
            url: URL to fetch
            
        Returns:
            BeautifulSoup object or None if failed
        """
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            time.sleep(self.delay)  # Respectful delay
            return BeautifulSoup(response.content, 'html.parser')
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None
    
    def search_poet(self, poet_name: str) -> List[Tuple[str, str]]:
        """
        Search for a poet's works
        
        Args:
            poet_name: Poet name (can be in Devanagari or English)
            
        Returns:
            List of (poem_title, poem_url) tuples
        """
        print(f"Searching for poet: {poet_name}")
        
        # Build search URL
        search_url = f"https://kavitakosh.org/kk/index.php?title={quote(poet_name)}"
        
        soup = self._fetch_page(search_url)
        if not soup:
            return []
        
        poems = []
        
        # Look for links to poems
        content_div = soup.find('div', {'id': 'mw-content-text'})
        if content_div:
            links = content_div.find_all('a', href=True)
            for link in links:
                href = link.get('href', '')
                if '/kk/index.php?title=' in href:
                    poem_title = link.text.strip()
                    poem_url = f"https://kavitakosh.org{href}" if href.startswith('/') else href
                    if poem_title and poem_title != poet_name:  # Skip the poet page itself
                        poems.append((poem_title, poem_url))
        
        print(f"Found {len(poems)} items for {poet_name}")
        return poems
    
    def download_poem(self, url: str, poet_name: str = "Unknown") -> Optional[Poem]:
        """
        Download and parse a single poem
        
        Args:
            url: URL of the poem
            poet_name: Name of the poet (for metadata)
            
        Returns:
            Poem object or None if failed
        """
        soup = self._fetch_page(url)
        if not soup:
            return None
        
        # Extract title
        title_elem = soup.find('h1', {'class': 'firstHeading'})
        title = title_elem.text.strip() if title_elem else "Unknown"
        
        # Extract content
        content_div = soup.find('div', {'id': 'mw-content-text'})
        if not content_div:
            return None
        
        # Get poem text
        poem_text = self._extract_poem_text(content_div)
        
        if not poem_text or len(poem_text.strip()) < 10:
            return None
        
        # Detect language
        language = self._detect_language(poem_text)
        
        # Extract categories
        categories = self._extract_categories(soup)
        
        # Create Poem object
        from datetime import datetime
        poem = Poem(
            title=title,
            author=poet_name,
            text=poem_text,
            language=language,
            url=url,
            categories=categories,
            download_time=datetime.now().isoformat()
        )
        
        self.poems_data.append(poem)
        self.stats[language] += 1
        
        return poem
    
    def _extract_poem_text(self, content_div) -> str:
        """Extract clean poem text from content div"""
        # Remove unwanted elements
        for elem in content_div.find_all(['script', 'style', 'sup']):
            elem.decompose()
        
        # Get all text, preserving line breaks
        lines = []
        for elem in content_div.find_all(['p', 'div']):
            text = elem.get_text(strip=True)
            if text and len(text) > 5:
                lines.append(text)
        
        return '\n'.join(lines)
    
    def _detect_language(self, text: str) -> str:
        """Detect language from script"""
        # Unicode ranges for various Indic scripts
        scripts = {
            'Devanagari': range(0x0900, 0x097F),
            'Bengali': range(0x0980, 0x09FF),
            'Gujarati': range(0x0A80, 0x0AFF),
            'Gurmukhi': range(0x0A00, 0x0A7F),
            'Kannada': range(0x0C80, 0x0CFF),
            'Malayalam': range(0x0D00, 0x0D7F),
            'Marathi': range(0x0900, 0x097F),  # Same as Devanagari
            'Odia': range(0x0B00, 0x0B7F),
            'Tamil': range(0x0B80, 0x0BFF),
            'Telugu': range(0x0C00, 0x0C7F),
            'Urdu': range(0x0600, 0x06FF),
        }
        
        detected = []
        for script_name, char_range in scripts.items():
            if any(ord(c) in char_range for c in text):
                detected.append(script_name)
        
        if detected:
            return ', '.join(detected)
        
        if any(ord(c) > 127 for c in text):
            return 'Non-Latin'
        
        return 'English/Latin'
    
    def _extract_categories(self, soup) -> List[str]:
        """Extract category links"""
        categories = []
        for link in soup.find_all('a', {'title': lambda x: x and x.startswith('Category:')}):
            cat = link.text.strip()
            if cat:
                categories.append(cat)
        return categories
    
    def download_poet_works(self, poet_name: str, max_poems: int = None) -> List[Poem]:
        """
        Download all works of a poet
        
        Args:
            poet_name: Poet name
            max_poems: Maximum number of poems to download
            
        Returns:
            List of Poem objects
        """
        poems_list = self.search_poet(poet_name)
        
        if max_poems:
            poems_list = poems_list[:max_poems]
        
        downloaded = []
        for title, url in poems_list:
            print(f"Downloading: {title}")
            poem = self.download_poem(url, poet_name)
            if poem:
                downloaded.append(poem)
        
        return downloaded
    
    def save_data(self):
        """Save all downloaded poems to disk"""
        # Save individual poem files
        poems_dir = self.output_dir / 'poems'
        poems_dir.mkdir(exist_ok=True)
        
        for i, poem in enumerate(self.poems_data):
            safe_filename = "".join(
                c for c in poem.title if c.isalnum() or c in (' ', '_', '-')
            ).rstrip()[:50]
            
            filename = poems_dir / f"{i:04d}_{safe_filename}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Title: {poem.title}\n")
                f.write(f"Author: {poem.author}\n")
                f.write(f"Language: {poem.language}\n")
                f.write(f"URL: {poem.url}\n")
                f.write(f"Categories: {', '.join(poem.categories)}\n")
                f.write("=" * 50 + "\n\n")
                f.write(poem.text)
        
        # Save metadata as JSONL
        metadata_file = self.output_dir / 'metadata.jsonl'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            for poem in self.poems_data:
                f.write(json.dumps(asdict(poem), ensure_ascii=False) + '\n')
        
        # Save statistics
        stats_file = self.output_dir / 'stats.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump({
                'total_poems': len(self.poems_data),
                'language_distribution': dict(self.stats),
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\nSaved {len(self.poems_data)} poems to {self.output_dir}")
        print(f"Language distribution: {dict(self.stats)}")
    
    def get_summary(self) -> Dict:
        """Get summary statistics"""
        return {
            'total_poems': len(self.poems_data),
            'languages': dict(self.stats),
            'average_poem_length': sum(len(p.text) for p in self.poems_data) / max(len(self.poems_data), 1),
        }


# Example usage for Jupyter notebooks or scripts
if __name__ == '__main__':
    # Initialize crawler
    crawler = KavitakoshCrawler(delay=1.5)
    
    # Download works of famous North Indian poets
    poets = [
        'कबीर',      # Kabir (Hindi/Urdu)
        'तुलसीदास',   # Tulsidas (Hindi)
        'सूरदास',     # Surdas (Hindi)
    ]
    
    for poet in poets:
        print(f"\n{'='*60}")
        print(f"Downloading works of: {poet}")
        print(f"{'='*60}")
        crawler.download_poet_works(poet, max_poems=10)
        time.sleep(2)  # Extra delay between poets
    
    # Save all data
    crawler.save_data()
    
    # Print summary
    print("\nCrawl Summary:")
    print(json.dumps(crawler.get_summary(), ensure_ascii=False, indent=2))
