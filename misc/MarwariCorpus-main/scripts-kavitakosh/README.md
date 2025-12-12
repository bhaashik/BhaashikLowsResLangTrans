# Kavita Kosh Data Crawler

Complete toolkit for crawling North Indian language poetry from [Kavita Kosh](https://kavitakosh.org) for NLP research.

## 📚 What's Included

- **kavitakosh_simple.py** - BeautifulSoup-based crawler (ideal for research & Jupyter)
- **kavitakosh_crawler.py** - Scrapy-based crawler (for large-scale production crawling)
- **usage_examples.py** - Ready-to-run examples for different use cases
- **KAVITAKOSH_GUIDE.md** - Comprehensive reference guide

## 🚀 Quick Start (2 minutes)

### Installation
```bash
pip install requests beautifulsoup4
```

### Download Poetry from a Specific Poet

```python
import kavitakosh_simple

# Initialize crawler
crawler = KavitakoshCrawler(output_dir='my_poetry', delay=1.5)

# Download Kabir's poems
crawler.download_poet_works('कबीर', max_poems=20)

# Save everything to disk
crawler.save_data()

# Check statistics
print(crawler.get_summary())
```

## 📖 What Languages Are Supported?

This crawler works with **North Indian languages** on Kavita Kosh:

- **Hindi** (हिन्दी) - कबीर, तुलसीदास, सूरदास
- **Marathi** (मराठी) - ज्ञानेश्वर, नामदेव
- **Bengali** (বাঙালি) - Rabindranath Tagore
- **Urdu** (اردو) - Mir, Ghalib
- **Sanskrit** (संस्कृत)
- And others in Devanagari and related scripts

## 💻 Usage Examples

### In Jupyter Notebooks
```python
# Cell 1: Import and setup
from kavitakosh_simple import KavitakoshCrawler
crawler = KavitakoshCrawler()

# Cell 2: Search for poems
poems = crawler.search_poet('तुलसीदास')
print(f"Found {len(poems)} items")

# Cell 3: Download and preview first poem
poem = crawler.download_poem(poems[0][1])
print(poem.text[:500])

# Cell 4: Batch download
crawler.download_poet_works('तुलसीदास', max_poems=15)

# Cell 5: Save and view stats
crawler.save_data()
import json
print(json.dumps(crawler.get_summary(), ensure_ascii=False, indent=2))
```

### Command Line
```bash
python usage_examples.py cli --poet कबीर --max 50 --output ./data
```

### Large-Scale Crawling (Scrapy)
```bash
# For production/large datasets
pip install scrapy
python -c "from kavitakosh_crawler import KavitakoshSpider; print('Ready for crawling')"
```

## 📊 Output Structure

```
my_poetry/
├── poems/
│   ├── 0001_कबीर_साखी.txt
│   ├── 0002_तुलसीदास_रामचरितमानस.txt
│   └── ...
├── metadata.jsonl    # One poem per line (JSON format)
└── stats.json        # Language distribution & totals
```

## 🔍 Key Features

✅ **Respectful Crawling**
- Configurable delays between requests
- Respects robots.txt
- Proper User-Agent headers

✅ **Language Detection**
- Automatically detects script (Devanagari, Urdu, Bengali, etc.)
- Useful for preprocessing

✅ **Metadata Extraction**
- Poem title, author, categories
- Language information
- Download timestamps

✅ **Research-Friendly**
- JSONL format for easy processing
- Works in Jupyter notebooks
- Simple Python API

✅ **Production-Ready**
- Scrapy-based crawler for large datasets
- Error handling and retries
- Progress tracking

## 🎯 Common Use Cases

### Case 1: Build a Hindi Poetry Corpus
```python
from kavitakosh_simple import KavitakoshCrawler

crawler = KavitakoshCrawler()

hindi_poets = ['कबीर', 'तुलसीदास', 'सूरदास', 'मीराबाई']

for poet in hindi_poets:
    crawler.download_poet_works(poet, max_poems=20)

crawler.save_data()
```

### Case 2: Language Detection & Classification
```python
import json

# Load all poems
with open('my_poetry/metadata.jsonl') as f:
    poems = [json.loads(line) for line in f]

# Group by language
by_language = {}
for p in poems:
    by_language.setdefault(p['language'], []).append(p)

for lang, ps in by_language.items():
    print(f"{lang}: {len(ps)} poems, {sum(x['content_length'] for x in ps)} total chars")
```

### Case 3: NLP Preprocessing
```python
from pathlib import Path

# Simple word frequency analysis
from collections import Counter
words = Counter()

for poem_file in Path('my_poetry/poems').glob('*.txt'):
    with open(poem_file) as f:
        content = f.read()
        words.update(content.split())

# Top terms
for word, count in words.most_common(20):
    print(f"{word}: {count}")
```

## 📋 Poetry Examples

Here are some North Indian poets in Kavita Kosh:

| Poet | Language | Period | Known For |
|------|----------|--------|-----------|
| कबीर (Kabir) | Hindi | 15th c. | Devotional, Sufi-influenced poems |
| तुलसीदास (Tulsidas) | Hindi | 16th c. | Ramacharitmanas, religious epics |
| सूरदास (Surdas) | Hindi | 16th c. | Krishna devotional poetry |
| मीराबाई (Meerabai) | Hindi | 16th c. | Bhakti movement, love poetry |
| रहीम (Rahim) | Hindi/Urdu | 16th c. | Wisdom poetry (doha), Islamic themes |

## ⚙️ Configuration

```python
# Crawler options
crawler = KavitakoshCrawler(
    output_dir='my_data',  # Where to save
    delay=1.5              # Seconds between requests (be respectful!)
)
```

### Delays (Important!)
- `delay=0.5` - Fast (only if site confirms stability)
- `delay=1.0` - Moderate (safe for most cases)
- `delay=2.0` - Conservative (recommended for bulk downloads)
- `delay=5.0` - Very conservative (for testing)

## 🔗 Integration with NLP Libraries

### spaCy (Hindi processing)
```python
import spacy
nlp = spacy.load("hi_core_news_sm")

for poem_file in Path('my_poetry/poems').glob('*.txt'):
    with open(poem_file) as f:
        doc = nlp(f.read())
        # Use spaCy for tokenization, NER, dependency parsing
```

### NLTK
```python
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

for poem_file in Path('my_poetry/poems').glob('*.txt'):
    with open(poem_file) as f:
        tokens = word_tokenize(f.read())
        # Process tokens
```

### Hugging Face Datasets
```python
from datasets import Dataset
import json

poems = []
with open('my_poetry/metadata.jsonl') as f:
    poems = [json.loads(line) for line in f]

ds = Dataset.from_list(poems)
```

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| SSL Certificate Error | Code handles gracefully; use delay >= 1.0 |
| No poems found | Verify poet name spelling (use Devanagari if possible) |
| Slow downloads | Reduce delay gradually (e.g., 1.5 → 1.0) |
| Encoding issues | All files use UTF-8 automatically |
| Incomplete downloads | Increase `max_poems` limit or retry manually |

## 📝 For Research Publications

If you use this crawler in academic work:

```bibtex
@software{kavitakosh_crawler,
  title={Kavita Kosh Crawler: North Indian Poetry Dataset},
  url={https://kavitakosh.org},
  year={2024},
  note={Retrieved from}
}

@online{kavitakosh,
  title={Kavita Kosh: Online Repository of Indian Literature},
  url={https://kavitakosh.org},
  organization={Samyak Lalit}
}
```

## ⚖️ License & Ethics

- **Kavita Kosh** content is openly available for research
- Always credit the source in publications
- Don't republish without attribution
- Respect the server (use appropriate delays)
- Ask permission for large-scale commercial use

## 🤝 Contributing

Found a bug? Want to improve the code?
- Modify the scripts as needed for your research
- Share improvements with colleagues
- Consider contributing back to the community

## 📚 Learn More

- **Full Guide**: See `KAVITAKOSH_GUIDE.md`
- **Examples**: See `usage_examples.py`
- **Website**: https://kavitakosh.org
- **NLP Resources**: https://docs.claude.com/en/docs/build-with-claude

---

**Happy researching! 🙏**

For questions or issues, refer to the `KAVITAKOSH_GUIDE.md` file or the comments in the Python files.
