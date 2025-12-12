#!/usr/bin/env python3
"""
Kavita Kosh Crawler - Usage Examples for NLP Research
This file contains examples for different use cases
"""

# ============================================================================
# OPTION 1: Simple script usage - Best for quick downloads
# ============================================================================

def example_basic_download():
    """Download poems from specific poets"""
    from crawler.kavitakosh_simple import KavitakoshCrawler
    
    crawler = KavitakoshCrawler(output_dir='kavitakosh_hindi', delay=1.5)
    
    # Download Kabir's works
    poems = crawler.download_poet_works('कबीर', max_poems=20)
    
    # Save to disk
    crawler.save_data()
    
    # Print stats
    print(crawler.get_summary())


# ============================================================================
# OPTION 2: Jupyter Notebook usage - Best for research and experimentation
# ============================================================================

def example_jupyter_workflow():
    """
    Code for Jupyter cells - interactive exploration
    """
    
    # Cell 1: Setup
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path.cwd()))
    
    from crawler.kavitakosh_simple import KavitakoshCrawler
    import json
    
    # Cell 2: Initialize crawler
    crawler = KavitakoshCrawler(output_dir='kavitakosh_data', delay=1.0)
    
    # Cell 3: Search for a specific poet
    poems = crawler.search_poet('तुलसीदास')
    for title, url in poems[:5]:
        print(f"• {title}")
    
    # Cell 4: Download a specific poem
    poem = crawler.download_poem(poems[0][1], 'तुलसीदास')
    print(f"Downloaded: {poem.title}")
    print(f"Language: {poem.language}")
    print(f"Content length: {poem.content_length} chars")
    print(f"\nFirst 300 chars:\n{poem.text[:300]}")
    
    # Cell 5: Download all poems by a poet
    crawler.download_poet_works('कबीर', max_poems=15)
    
    # Cell 6: View statistics
    print(json.dumps(crawler.get_summary(), ensure_ascii=False, indent=2))
    
    # Cell 7: Save data
    crawler.save_data()


# ============================================================================
# OPTION 3: Advanced - Filtering by language or custom logic
# ============================================================================

def example_language_filtering():
    """Download and filter by language"""
    from crawler.kavitakosh_simple import KavitakoshCrawler
    
    crawler = KavitakoshCrawler(delay=1.5)
    
    # Poets in different languages
    north_indian_poets = {
        'Hindi': ['कबीर', 'तुलसीदास', 'सूरदास', 'मीराबाई'],
        'Marathi': ['ज्ञानेश्वर', 'नामदेव'],
        'Bengali': ['রবীন্দ্রনাথ ঠাকুর'],
        'Urdu': ['मिर', 'ग़ालिब'],
    }
    
    for language, poets in north_indian_poets.items():
        print(f"\n{'='*60}")
        print(f"Downloading {language} poetry")
        print(f"{'='*60}")
        
        for poet in poets:
            try:
                print(f"Downloading: {poet}")
                poems = crawler.download_poet_works(poet, max_poems=10)
                print(f"  → Downloaded {len(poems)} poems")
            except Exception as e:
                print(f"  → Error: {e}")
    
    # Filter and analyze
    devanagari_poems = [p for p in crawler.poems_data if 'Devanagari' in p.language]
    print(f"\nTotal Devanagari poems: {len(devanagari_poems)}")
    
    # Save only Devanagari poems
    crawler.poems_data = devanagari_poems
    crawler.save_data()


# ============================================================================
# OPTION 4: For AI Agents - Integration with agent frameworks
# ============================================================================

def example_with_ai_agent():
    """
    Example for integration with LangChain/other AI agent frameworks
    """
    
    from crawler.kavitakosh_simple import KavitakoshCrawler
    
    class KavitakoshTool:
        """Wrapper for use as an AI agent tool"""
        
        def __init__(self):
            self.crawler = KavitakoshCrawler(delay=1.0)
        
        def search_poems_by_poet(self, poet_name: str) -> list:
            """Search poems by poet name"""
            results = self.crawler.search_poet(poet_name)
            return [
                {
                    'title': title,
                    'url': url,
                    'preview': 'Click to download'
                }
                for title, url in results[:10]
            ]
        
        def get_poem_text(self, url: str) -> dict:
            """Get full poem text"""
            poem = self.crawler.download_poem(url)
            if poem:
                return {
                    'title': poem.title,
                    'author': poem.author,
                    'language': poem.language,
                    'text': poem.text,
                    'categories': poem.categories,
                }
            return {'error': 'Failed to download poem'}
        
        def analyze_corpus(self) -> dict:
            """Get statistics about downloaded corpus"""
            return self.crawler.get_summary()
    
    # Usage in agent
    tool = KavitakoshTool()
    
    # Agent can now call these methods
    results = tool.search_poems_by_poet('कबीर')
    if results:
        poem = tool.get_poem_text(results[0]['url'])
        print(poem)


# ============================================================================
# OPTION 5: Command-line usage with arguments
# ============================================================================

def example_cli_interface():
    """Run from command line: python usage_examples.py --poet कबीर --max 20"""
    
    import argparse
    from crawler.kavitakosh_simple import KavitakoshCrawler
    
    parser = argparse.ArgumentParser(
        description='Download poetry from Kavita Kosh'
    )
    parser.add_argument('--poet', type=str, required=True, help='Poet name')
    parser.add_argument('--max', type=int, default=10, help='Max poems to download')
    parser.add_argument('--output', type=str, default='kavitakosh_data', help='Output directory')
    parser.add_argument('--delay', type=float, default=1.5, help='Delay between requests')
    
    args = parser.parse_args()
    
    crawler = KavitakoshCrawler(output_dir=args.output, delay=args.delay)
    poems = crawler.download_poet_works(args.poet, max_poems=args.max)
    crawler.save_data()
    
    print(f"Downloaded {len(poems)} poems")


# ============================================================================
# OPTION 6: NLP research - Processing downloaded poems
# ============================================================================

def example_nlp_processing():
    """
    Process downloaded poems for NLP research
    """
    from pathlib import Path
    import json
    
    # Read all poem metadata
    metadata_file = Path('kavitakosh_data/metadata.jsonl')
    
    poems_by_language = {}
    total_words = 0
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        for line in f:
            poem_data = json.loads(line)
            language = poem_data.get('language', 'Unknown')
            
            if language not in poems_by_language:
                poems_by_language[language] = []
            
            poems_by_language[language].append(poem_data)
            total_words += len(poem_data['text'].split())
    
    print("NLP Research Corpus Statistics:")
    print(f"Total poems: {sum(len(v) for v in poems_by_language.values())}")
    print(f"Total words: {total_words}")
    print("\nLanguage distribution:")
    for lang, poems in poems_by_language.items():
        avg_length = sum(p['content_length'] for p in poems) / len(poems)
        print(f"  {lang}: {len(poems)} poems, avg {avg_length:.0f} chars")
    
    # Example: Tokenization for a specific language
    poems_dir = Path('kavitakosh_data/poems')
    
    # Simple word frequency for Hindi poetry
    from collections import Counter
    hindi_words = Counter()
    
    for poem_file in poems_dir.glob('*.txt'):
        with open(poem_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # Simple split (for production, use proper Hindi tokenizer)
            words = content.split()
            hindi_words.update(words)
    
    print("\nTop 20 most frequent words:")
    for word, count in hindi_words.most_common(20):
        print(f"  {word}: {count}")


# ============================================================================
# Installation and Dependencies
# ============================================================================

"""
Installation:
    pip install requests beautifulsoup4 scrapy

For Jupyter notebooks:
    pip install jupyter
    jupyter notebook

For NLP research, also consider:
    pip install nltk spacy transformers datasets
    
    For Hindi specifically:
    pip install indic-nlp-library marisa-trie
"""

# ============================================================================
# Running the Examples
# ============================================================================

if __name__ == '__main__':
    import sys
    
    examples = {
        'basic': example_basic_download,
        'language': example_language_filtering,
        'cli': example_cli_interface,
        'nlp': example_nlp_processing,
    }
    
    if len(sys.argv) > 1:
        example_name = sys.argv[1]
        if example_name in examples:
            examples[example_name]()
        else:
            print(f"Unknown example: {example_name}")
            print(f"Available: {', '.join(examples.keys())}")
    else:
        print("Usage: python usage_examples.py <example_name>")
        print(f"Available examples: {', '.join(examples.keys())}")
        print("\nRunning basic example...")
        example_basic_download()
