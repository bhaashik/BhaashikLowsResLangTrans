# Kavita Kosh Crawler - Quick Reference Guide

## Overview
This guide helps you crawl North Indian language poetry from https://kavitakosh.org for NLP research.

## Files Provided

1. **kavitakosh_simple.py** - Simple BeautifulSoup-based crawler (recommended for research)
2. **kavitakosh_crawler.py** - Advanced Scrapy-based crawler (for large-scale crawling)
3. **usage_examples.py** - Examples for different use cases

## Quick Start

### Installation
```bash
pip install requests beautifulsoup4
# For Scrapy crawler:
pip install scrapy
```

### Basic Usage (5 minutes)
```python
from kavitakosh_simple import KavitakoshCrawler

# Create crawler
crawler = KavitakoshCrawler(output_dir='my_poems', delay=1.5)

# Download Kabir's poems
poems = crawler.download_poet_works('कबीर', max_poems=20)

# Save to disk
crawler.save_data()

# Get statistics
print(crawler.get_summary())
```

## Common Use Cases

### Use Case 1: Download a Specific Poet's Works
```python
from kavitakosh_simple import KavitakoshCrawler

crawler = KavitakoshCrawler()

# Search for poems
poems_list = crawler.search_poet('तुलसीदास')
print(f"Found {len(poems_list)} items")

# Download first 10 poems
for title, url in poems_list[:10]:
    poem = crawler.download_poem(url, 'तुलसीदास')
    if poem:
        print(f"Downloaded: {poem.title}")

crawler.save_data()
```

### Use Case 2: Download Multiple Poets
```python
poets = ['कबीर', 'तुलसीदास', 'सूरदास', 'मीराबाई']

for poet in poets:
    crawler.download_poet_works(poet, max_poems=15)
    # Built-in delay between requests

crawler.save_data()
```

### Use Case 3: Filter by Language
```python
# Download and filter Devanagari scripts
crawler.download_poet_works('कबीर', max_poems=20)

# Filter only Devanagari poems
devanagari_poems = [
    p for p in crawler.poems_data 
    if 'Devanagari' in p.language
]

# Save filtered set
crawler.poems_data = devanagari_poems
crawler.save_data()
```

### Use Case 4: Jupyter Notebook Usage
```python
# Cell 1
from kavitakosh_simple import KavitakoshCrawler
import json

# Cell 2
crawler = KavitakoshCrawler()

# Cell 3: Search and preview
poems = crawler.search_poet('कबीर')
print(f"Found {len(poems)} poems/pages")

# Cell 4: Download a specific poem
if poems:
    poem = crawler.download_poem(poems[0][1])
    print(poem.text[:500])  # Preview

# Cell 5: Batch download
for poet in ['कबीर', 'तुलसीदास']:
    crawler.download_poet_works(poet, max_poems=10)

# Cell 6: Save and analyze
crawler.save_data()
summary = crawler.get_summary()
print(json.dumps(summary, ensure_ascii=False, indent=2))
```

### Use Case 5: Command Line
```bash
# Download Kabir's works
python usage_examples.py cli --poet कबीर --max 50 --output ./data

# Or run directly
python -c "
from kavitakosh_simple import KavitakoshCrawler
c = KavitakoshCrawler()
c.download_poet_works('कबीर', max_poems=20)
c.save_data()
"
```

### Use Case 6: NLP Processing
```python
from pathlib import Path
import json
from collections import Counter

# Read metadata
with open('kavitakosh_data/metadata.jsonl') as f:
    poems = [json.loads(line) for line in f]

# Language distribution
by_lang = {}
for p in poems:
    by_lang.setdefault(p['language'], []).append(p)

for lang, ps in by_lang.items():
    print(f"{lang}: {len(ps)} poems")

# Word frequency (simple example)
words = Counter()
for poem_file in Path('kavitakosh_data/poems').glob('*.txt'):
    with open(poem_file) as f:
        content = f.read()
        words.update(content.split())

print("Top 20 words:")
for word, count in words.most_common(20):
    print(f"  {word}: {count}")
```

## Available North Indian Poets

### Hindi
- कबीर (Kabir)
- तुलसीदास (Tulsidas)
- सूरदास (Surdas)
- मीराबाई (Meerabai)
- रहीम (Rahim)
- सुमित्रानंदन पंत (Sumitranandan Pant)

### Marathi
- ज्ञानेश्वर (Jnaneswari)
- नामदेव (Namdev)

### Bengali
- রবীন্দ্রনাথ ঠাকুর (Rabindranath Tagore)

### Urdu/Persianized
- मिर (Mir)
- ग़ालिब (Ghalib)

## Output Directory Structure
```
kavitakosh_data/
├── poems/
│   ├── 0001_कबीर_साखी.txt
│   ├── 0002_तुलसीदास_रामचरितमानस.txt
│   └── ...
├── metadata.jsonl          # Metadata for each poem (one per line)
└── stats.json              # Overall statistics
```

## Poem Metadata Fields
```json
{
  "title": "साखी",
  "author": "कबीर",
  "text": "...",
  "language": "Devanagari",
  "url": "https://...",
  "categories": ["devotional", "folk"],
  "download_time": "2024-11-09T10:30:00",
  "content_length": 1250
}
```

## Performance Tips

### For Large-Scale Downloads
1. Use the Scrapy crawler (`kavitakosh_crawler.py`)
2. Increase `concurrent_requests` gradually
3. Reduce `delay` after confirming stability

### For Research
1. Start with `kavitakosh_simple.py` - easier to debug
2. Use Jupyter notebooks for interactive exploration
3. Save metadata as JSONL for easy processing

### Rate Limiting
- Default delay: 1.5 seconds between requests
- Respects robots.txt
- Change via `delay` parameter:
```python
crawler = KavitakoshCrawler(delay=2.0)  # 2 seconds between requests
```

## Troubleshooting

### SSL Certificate Error
```
Solution: The site may have SSL issues. Code handles this gracefully.
```

### No poems found
```python
# Check if poet exists on the site
poems_list = crawler.search_poet('कबीर')
if not poems_list:
    print("Poet not found or page structure changed")
```

### Slow downloads
```python
# Reduce delay if site is stable
crawler = KavitakoshCrawler(delay=0.5)

# Or use Scrapy for parallel downloads:
# python kavitakosh_crawler.py
```

### Encoding issues
```python
# All code uses UTF-8 encoding by default
# For manual checks:
with open('file.txt', encoding='utf-8') as f:
    content = f.read()
```

## Ethical Considerations

✅ **Good practices:**
- Respect rate limits (delay between requests)
- Identify yourself in User-Agent
- Cache/store downloaded content
- Credit Kavita Kosh in publications

❌ **Avoid:**
- Aggressive/rapid requests
- Republishing without attribution
- Modifying copyrighted content

## Integration with NLP Pipelines

### With spaCy
```python
import spacy
from pathlib import Path

nlp = spacy.load("hi_core_news_sm")  # Hindi model

for poem_file in Path('kavitakosh_data/poems').glob('*.txt'):
    with open(poem_file) as f:
        text = f.read()
        doc = nlp(text)
        # Process tokens, entities, etc.
```

### With Hugging Face Datasets
```python
from datasets import Dataset
import json

# Load metadata
poems = []
with open('kavitakosh_data/metadata.jsonl') as f:
    poems = [json.loads(line) for line in f]

# Create HuggingFace dataset
dataset = Dataset.from_list(poems)
dataset.push_to_hub("your-org/kavitakosh-hindi")
```

### With NLTK
```python
from nltk.tokenize import sent_tokenize
import os

for poem_file in os.listdir('kavitakosh_data/poems'):
    with open(f'kavitakosh_data/poems/{poem_file}') as f:
        text = f.read()
        sentences = sent_tokenize(text)
        # Process sentences
```

## Advanced: Scrapy for Production

For large-scale production crawling:

```bash
python kavitakosh_crawler.py
```

This will:
- Crawl more efficiently
- Handle errors gracefully
- Save structured data
- Respect rate limits automatically

## Contributing

If you improve the crawler, consider contributing back:
1. Fix bugs or add features
2. Test with different poets/languages
3. Share improvements with the community

## Citation

If you use this crawler in research, cite as:

```
@software{kavitakosh_crawler,
  title={Kavita Kosh Crawler for North Indian Poetry},
  url={https://kavitakosh.org},
  year={2024}
}
```

## Support

- **Website**: https://kavitakosh.org
- **GitHub**: Check for issues/updates
- **Questions**: Modify code as needed for your research

---

**Last updated**: November 2024
