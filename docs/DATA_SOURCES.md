# Data Sources for Monolingual Examples

Complete guide to obtaining monolingual data for all supported low-resource Indic languages.

## Summary Table

| Language | Code | Primary Source | Sentences Available | Status |
|----------|------|----------------|---------------------|--------|
| **Bhojpuri** | bho | Local corpus | ~Thousands | ✅ Available |
| **Magahi** | mag | Local corpus | ~Thousands | ✅ Available |
| **Maithili** | mai | Sangrah dataset | ~Millions | ✅ Available |
| **Nepali** | nep | Sangrah dataset | ~Millions | ✅ Available |
| **Awadhi** | awa | VarDial 2018 | ~15,000 | ⚠️ Need to download |
| **Braj** | bra | VarDial 2018 | ~15,000 | ⚠️ Need to download |
| **Marwari** | mwr | Need to find | ? | ❌ No source yet |

## Detailed Sources

### 1. Sangrah Dataset (Maithili, Nepali)

**What is it**: AI4Bharat's largest cleaned Indic language corpus with 251B tokens across 22 scheduled languages.

**Languages available**:
- Maithili (mai) - ✅
- Nepali (nep) - ✅
- **NOT** Marwari, Awadhi, or Braj

**How to access**:
```bash
# Install dependencies
pip install datasets huggingface-hub

# Authenticate
huggingface-cli login

# Run extraction script
python scripts/extract_examples_from_sangrah.py
```

**Links**:
- [Sangrah Dataset on HuggingFace](https://huggingface.co/datasets/ai4bharat/sangraha)
- [AI4Bharat Official Site](https://ai4bharat.iitm.ac.in/)

---

### 2. VarDial 2018 Dataset (Awadhi, Braj)

**What is it**: Indo-Aryan Language Identification Shared Task dataset with 15,000 sentences per language in the literature domain.

**Languages available**:
- Awadhi (awa) - ✅
- Braj (bra) - ✅
- Bhojpuri (bho) - ✅ (but you already have local data)
- Magahi (mag) - ✅ (but you already have local data)
- Hindi (hin)

**How to access**:
```bash
# Clone the dataset
git clone https://github.com/kmi-linguistics/vardial2018.git

# Extract examples
python scripts/extract_vardial_examples.py
```

**Output**:
- `universal_translate/data/examples/hi_to_awa_monolingual.json` (100 examples)
- `universal_translate/data/examples/hi_to_bra_monolingual.json` (100 examples)

**Links**:
- [VarDial 2018 GitHub](https://github.com/kmi-linguistics/vardial2018)
- [COLING 2018 Paper](https://aclanthology.org/W18-3905/)

---

### 3. Local Corpus (Bhojpuri, Magahi, Maithili)

**What is it**: PV-BMM Public Data - Bhojpuri, Magahi, and Maithili linguistic resources.

**Location**: `input/PV-BMM-Public-Data/Bhojpuri-Magahi-and-Maithili-Linguistic-Resources/`

**Languages available**:
- Bhojpuri (bho) - ✅
- Magahi (mag) - ✅
- Maithili (mai) - ✅

**How to access**: Already available in your repository.

---

### 4. Marwari - Sources to Explore

Marwari is **not** in Sangrah or VarDial 2018. Here are potential sources:

#### Option A: LDCIL (Linguistic Data Consortium for Indian Languages)
- Website: [https://data.ldcil.org/](https://data.ldcil.org/)
- May have Marwari corpora
- Requires registration and possibly fees

#### Option B: Indian Language Corpora Initiative (ILCI)
- Multi-lingual parallel corpus project
- Check: [https://tdil-dc.in/ilci/](https://tdil-dc.in/ilci/)

#### Option C: Web Scraping
Collect from Marwari websites:
- Wikipedia Marwari: `mwr.wikipedia.org`
- News sites in Marwari
- Literature websites

Example script:
```python
import requests
from bs4 import BeautifulSoup

# Scrape Marwari Wikipedia
url = "https://mwr.wikipedia.org/wiki/मुख्य_पान"
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
# Extract paragraphs...
```

#### Option D: Manual Collection
- Marwari literature books
- Folk songs (लोकगीत)
- Proverbs and sayings
- Social media content

#### Option E: Use Hindi as Proxy
For initial testing, use Hindi examples:
```bash
# Temporarily use Hindi examples for Marwari
cp universal_translate/data/examples/hi_to_bho_monolingual.json \
   universal_translate/data/examples/hi_to_mwr_monolingual.json
```

Then manually edit to be more Marwari-like or collect 50-100 Marwari sentences manually.

---

## How to Populate Example Files

### Step 1: Download Required Datasets

```bash
# For Maithili and Nepali
pip install datasets huggingface-hub
huggingface-cli login
python scripts/extract_examples_from_sangrah.py

# For Awadhi and Braj
git clone https://github.com/kmi-linguistics/vardial2018.git
python scripts/extract_vardial_examples.py
```

### Step 2: Verify Examples

```bash
# Check number of examples
cat universal_translate/data/examples/hi_to_awa_monolingual.json | jq '.num_examples'
cat universal_translate/data/examples/hi_to_bra_monolingual.json | jq '.num_examples'
cat universal_translate/data/examples/hi_to_mai_monolingual.json | jq '.num_examples'
cat universal_translate/data/examples/hi_to_nep_monolingual.json | jq '.num_examples'

# View first 3 examples
cat universal_translate/data/examples/hi_to_awa_monolingual.json | jq '.examples[:3]'
```

### Step 3: Manual Collection for Marwari

If no corpus is available, collect manually:

```python
# Create Marwari examples manually
import json

examples = {
    "source": "Manual collection",
    "language": "mwr",
    "language_name": "Marwari",
    "num_examples": 50,
    "examples": [
        {"text": "म्हैं घर जाऊं छूं।", "language": "mwr", "language_name": "Marwari"},
        {"text": "थे कठै जावो छो?", "language": "mwr", "language_name": "Marwari"},
        # Add 48 more...
    ]
}

with open('universal_translate/data/examples/hi_to_mwr_monolingual.json', 'w', encoding='utf-8') as f:
    json.dump(examples, f, ensure_ascii=False, indent=2)
```

---

## Quality Requirements

For good prompt caching and translation quality:

### Minimum Requirements
- **Number**: At least 50 examples
- **Length**: 20-200 characters per sentence
- **Diversity**: Different sentence structures, tenses, topics
- **Quality**: Natural, grammatically correct sentences

### Recommended
- **Number**: 100-150 examples
- **Domains**: Mix of conversational, literary, news
- **Coverage**: Common grammar patterns and vocabulary

---

## Troubleshooting

### Sangrah: Language Not Found

```python
# Check available languages
from datasets import load_dataset
dataset = load_dataset("ai4bharat/sangraha", streaming=True)
print(dataset.column_names)  # See what's available
```

### VarDial: File Not Found

```bash
# List directory structure
ls -R vardial2018/

# Files might be in different location
find vardial2018/ -name "*AWA*" -o -name "*BRA*"
```

### Insufficient Examples

If VarDial has <100 good sentences:
1. Lower quality filter thresholds
2. Supplement with other sources
3. Manually add more examples

---

## References

- [Sangrah Dataset](https://huggingface.co/datasets/ai4bharat/sangraha)
- [VarDial 2018](https://github.com/kmi-linguistics/vardial2018)
- [AI4Bharat IndicNLP Catalog](https://github.com/AI4Bharat/indicnlp_catalog)
- [LDCIL Data Resources](https://data.ldcil.org/)
- [ILCI Corpus](https://tdil-dc.in/ilci/)

---

## Current Status

After running all extraction scripts, you should have:

```
universal_translate/data/examples/
├── hi_to_awa_monolingual.json  # 100 examples from VarDial 2018
├── hi_to_bho_monolingual.json  # Existing (local corpus)
├── hi_to_bra_monolingual.json  # 100 examples from VarDial 2018
├── hi_to_mag_monolingual.json  # Existing (local corpus)
├── hi_to_mai_monolingual.json  # 100 examples from Sangrah
├── hi_to_mwr_monolingual.json  # ❌ Need to create manually or find source
└── hi_to_nep_monolingual.json  # 100 examples from Sangrah
```

## Next Steps

1. **Download VarDial 2018**: Get Awadhi and Braj examples
2. **Extract from Sangrah**: Get Maithili and Nepali examples
3. **Find Marwari source**: Explore LDCIL or manual collection
4. **Verify quality**: Check that examples are natural and diverse
5. **Test translation**: Run example scripts with real data
