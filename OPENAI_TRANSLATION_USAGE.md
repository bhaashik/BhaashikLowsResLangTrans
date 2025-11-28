# OpenAI Translation with Prompt Caching - Usage Guide

## Overview

This guide explains how to translate Hindi text to low-resource Indic languages (Bhojpuri, Magahi, Maithili) using OpenAI's GPT-4o-mini with prompt caching for cost optimization.

## Key Features

1. **Prompt Caching**: Uses monolingual examples that get cached by OpenAI (50% cost savings on cached tokens)
2. **Monolingual Examples**: 50 representative sentences per language from PV-BMM-Public-Data
3. **Directory Structure Preservation**: Maintains exact directory/file structure from input to output
4. **Batch Processing**: Efficient translation of multiple files

## Setup

### 1. Install Dependencies

```bash
pip install openai pyyaml
```

### 2. Set OpenAI API Key

```bash
export OPENAI_API_KEY="your_api_key_here"
```

### 3. Prepare Monolingual Examples

Examples have already been created in:
- `universal_translate/data/examples/hi_to_bho_monolingual.json` (Bhojpuri)
- `universal_translate/data/examples/hi_to_mag_monolingual.json` (Magahi)
- `universal_translate/data/examples/hi_to_mai_monolingual.json` (Maithili)

Each contains 50 diverse sentences sampled from the full corpus.

## Usage

### Basic Translation

Translate Hindi files to Bhojpuri:

```bash
python scripts/translate_hindi_to_lowres_openai.py --target-lang bho
```

### All Supported Languages

```bash
# Bhojpuri
python scripts/translate_hindi_to_lowres_openai.py --target-lang bho

# Magahi
python scripts/translate_hindi_to_lowres_openai.py --target-lang mag

# Maithili
python scripts/translate_hindi_to_lowres_openai.py --target-lang mai
```

### Options

```bash
# Dry run (show what would be done)
python scripts/translate_hindi_to_lowres_openai.py --target-lang bho --dry-run

# Cost estimate only
python scripts/translate_hindi_to_lowres_openai.py --target-lang bho --estimate-only

# Limit number of files (for testing)
python scripts/translate_hindi_to_lowres_openai.py --target-lang bho --max-files 5

# Custom batch size (sentences per API call)
python scripts/translate_hindi_to_lowres_openai.py --target-lang bho --batch-size 20

# Custom input/output directories
python scripts/translate_hindi_to_lowres_openai.py \
  --target-lang bho \
  --input-dir input/converted/Hindi/plain-text/by_domain \
  --output-dir output/Bhojpuri/plain-text/by_domain
```

## How It Works

### 1. Monolingual Examples as Context

The system loads 50 monolingual sentences in the target language:

```yaml
# From config/prompts/hi_to_bho_openai.yaml
examples_file: "../../data/examples/hi_to_bho_monolingual.json"
use_prompt_caching: true
```

These sentences are included in the system prompt to:
- Demonstrate natural language patterns
- Provide vocabulary and grammar context
- Enable OpenAI's prompt caching

### 2. Prompt Caching Benefits

**First API Call:**
- System prompt (with 50 examples): ~5,000 tokens
- Cost: Full input cost

**Subsequent API Calls:**
- System prompt: Cached (50% cost reduction)
- Only new text is charged at full rate

**Example Cost Savings:**

```
Without caching (1000 sentences):
  Input: 5,000 tokens × 1000 = 5M tokens × $0.15/1M = $0.75
  Output: ~50,000 tokens × $0.60/1M = $0.03
  Total: $0.78

With caching (1000 sentences):
  First call: 5,000 tokens × $0.15/1M = $0.00075
  Remaining 999: 5,000 tokens × $0.075/1M × 999 = $0.37
  Output: ~50,000 tokens × $0.60/1M = $0.03
  Total: $0.40 (49% savings!)
```

### 3. Directory Structure Recreation

Input structure:
```
input/converted/Hindi/plain-text/by_file/
├── AGRICULTURE/
│   ├── file1.txt
│   └── file2.txt
├── CONVERSATIONAL/
│   ├── file3.txt
│   └── file4.txt
└── ...
```

Output structure (identical):
```
output/Bhojpuri/plain-text/by_file/
├── AGRICULTURE/
│   ├── file1.txt  (translated)
│   └── file2.txt  (translated)
├── CONVERSATIONAL/
│   ├── file3.txt  (translated)
│   └── file4.txt  (translated)
└── ...
```

## Cost Estimation

Before translating, estimate costs:

```bash
python scripts/translate_hindi_to_lowres_openai.py \
  --target-lang bho \
  --estimate-only
```

Output:
```
Cost Estimate:
  Files: 1432
  Sentences: 66,317
  Estimated input cost: $2.15
  Estimated output cost: $1.98
  Total estimated cost: $4.13
  Cost per sentence: $0.000062

With caching benefits:
  System tokens: ~5000 tokens
  First request: Full cost
  Subsequent: ~50% savings on system prompt
```

## Configuration Files

### Prompt Configuration

Each language has its own prompt configuration:

**Bhojpuri** (`config/prompts/hi_to_bho_openai.yaml`):
```yaml
name: "hindi_to_bhojpuri_openai"
description: "Hindi to Bhojpuri translation..."

system_prompt: |
  You are an expert translator specializing in Hindi to Bhojpuri...
  [Guidelines and linguistic features]
  Below are examples of natural Bhojpuri sentences:

user_prompt: |
  Translate the following Hindi text to Bhojpuri:
  ${text}

examples_file: "../../data/examples/hi_to_bho_monolingual.json"
use_prompt_caching: true

parameters:
  temperature: 0.3
  max_tokens: 2048
```

### Example Data Format

Monolingual examples (`data/examples/hi_to_bho_monolingual.json`):
```json
{
  "examples": [
    {
      "text": "राम घर जात बाड़े।",
      "language": "bho",
      "language_name": "Bhojpuri",
      "note": "Monolingual example..."
    },
    ...
  ],
  "metadata": {
    "language": "bho",
    "num_examples": 50,
    "source": "PV-BMM-Public-Data",
    "purpose": "OpenAI prompt caching"
  }
}
```

## Python API Usage

For programmatic usage:

```python
from pathlib import Path
from universal_translate import TranslationRequest, TranslationUnit
from universal_translate.providers import OpenAIProvider
from universal_translate.prompts import PromptManager

# Load prompt configuration
prompt_mgr = PromptManager("universal_translate/config/prompts/hi_to_bho_openai.yaml")

# Create provider
provider = OpenAIProvider(
    model="gpt-4o-mini",
    prompt_manager=prompt_mgr,
    use_caching=True
)

# Read Hindi sentences
with open("input.txt", "r", encoding="utf-8") as f:
    sentences = [line.strip() for line in f if line.strip()]

# Create translation request
units = [TranslationUnit(text=text, index=i) for i, text in enumerate(sentences)]
request = TranslationRequest(units=units, src_lang="hi", tgt_lang="bho")

# Translate
with provider:
    response = provider.translate_sync(request)

# Write translations
with open("output.txt", "w", encoding="utf-8") as f:
    for result in sorted(response.results, key=lambda x: x.index):
        f.write(result.translation + "\n")

print(f"Cost: ${response.total_cost:.2f}")
```

## Monitoring Translation

### Real-time Progress

The script shows progress for each file:

```
[1/1432] AGRICULTURE/file1.txt
  Sentences: 50
  Translating batch 1-10/50... Done (cost: $0.0023)
  Translating batch 11-20/50... Done (cost: $0.0018)
  ...
  ✓ Written: output/Bhojpuri/.../file1.txt
```

### Final Summary

```
TRANSLATION SUMMARY
======================================================================
Files processed:       1432
Files skipped:         0
Sentences translated:  66,317
Total cost:            $3.85 USD
Average cost/sentence: $0.000058 USD
======================================================================
```

## Best Practices

### 1. Start Small
Test with a few files first:
```bash
python scripts/translate_hindi_to_lowres_openai.py \
  --target-lang bho \
  --max-files 5
```

### 2. Use Cost Estimation
Always estimate before large runs:
```bash
python scripts/translate_hindi_to_lowres_openai.py \
  --target-lang bho \
  --estimate-only
```

### 3. Optimize Batch Size
- Larger batches: Fewer API calls, better caching
- Smaller batches: Faster failure recovery
- Recommended: 10-20 sentences per batch

### 4. Monitor Costs
Check your OpenAI usage dashboard regularly.

## Troubleshooting

### Error: "OpenAI API key not provided"
```bash
export OPENAI_API_KEY="sk-..."
```

### Error: "openai package not installed"
```bash
pip install openai
```

### Rate Limiting
If you hit rate limits, reduce batch size or add delays.

### Memory Issues
For very large files, process in smaller batches:
```bash
python scripts/translate_hindi_to_lowres_openai.py \
  --target-lang bho \
  --batch-size 5
```

## Expected Costs

Based on GPT-4o-mini pricing (as of Dec 2025):
- Input: $0.15 per 1M tokens
- Output: $0.60 per 1M tokens
- Cached input: $0.075 per 1M tokens (50% discount)

**For translating all Hindi plain-text by_file (66,317 sentences):**
- Without caching: ~$7-8 USD
- With caching: ~$3.5-4.5 USD

**Cost per language:**
- Bhojpuri: ~$4 USD
- Magahi: ~$4 USD
- Maithili: ~$4 USD

**Total for all 3 languages: ~$12 USD**

## Verification

After translation, verify using the JSON mappings:

```python
import json

# Load mapping
with open("input/converted/Hindi/mappings/hindi_by_file_mappings.json") as f:
    mapping = json.load(f)

# Check sentence alignment
# Source sentence 10 in AGRICULTURE/file1.txt should align with
# translation sentence 10 in output/Bhojpuri/plain-text/by_file/AGRICULTURE/file1.txt
```

## Next Steps

1. Translate to all three languages
2. Evaluate translation quality on samples
3. Create parallel corpora by aligning with original Hindi
4. Use for downstream NLP tasks (MT training, POS tagging, etc.)

## Support

For issues:
1. Check OpenAI API key is set
2. Verify input files exist
3. Check disk space for output
4. Review error messages in output

For questions about the translation system, see:
- `universal_translate/README.md`
- `TRANSLATION_PACKAGE_DESIGN.md`
