# Quick Start: Translate Hindi to Low-Resource Languages with GPT-5-nano

## TL;DR

Translate 66,317 Hindi sentences to Bhojpuri, Magahi, or Maithili using OpenAI's GPT-5-nano with prompt caching for ~$2-3 per language.

```bash
# Set API key
export OPENAI_API_KEY="your_key"

# Translate to Bhojpuri (estimated cost: $2-3 USD)
python scripts/translate_hindi_to_lowres_openai.py --target-lang bho

# Or to all three languages
for lang in bho mag mai; do
    python scripts/translate_hindi_to_lowres_openai.py --target-lang $lang
done
```

## What You Get

**Input**: `input/converted/Hindi/plain-text/by_file/` (66,317 sentences in 1,432 files)

**Output**: `output/{Language}/plain-text/by_file/` (exact same structure)
- Bhojpuri translations
- Magahi translations
- Maithili translations

**Cost**: ~$2-3 per language with GPT-5-nano + prompt caching

## Setup (One-Time)

### 1. Install Dependencies

```bash
pip install openai pyyaml
```

### 2. Set API Key

```bash
export OPENAI_API_KEY="sk-..."
```

Or add to `.env`:
```bash
echo 'OPENAI_API_KEY=sk-...' >> .env
```

### 3. Verify Examples Are Ready

Already created for you:
```bash
ls universal_translate/data/examples/hi_to_*_monolingual.json
```

You should see:
- `hi_to_bho_monolingual.json` (50 Bhojpuri sentences)
- `hi_to_mag_monolingual.json` (50 Magahi sentences)
- `hi_to_mai_monolingual.json` (50 Maithili sentences)

## Usage

### Test Run (5 files, free/cheap)

```bash
python scripts/translate_hindi_to_lowres_openai.py \
  --target-lang bho \
  --max-files 5
```

### Cost Estimate

```bash
python scripts/translate_hindi_to_lowres_openai.py \
  --target-lang bho \
  --estimate-only
```

Expected output:
```
Cost Estimate:
  Files: 1432
  Sentences: 66,317
  Total estimated cost: $2.65 USD
  Cost per sentence: $0.000040

With caching benefits:
  First request: Full cost
  Subsequent: ~50% savings on system prompt
```

### Full Translation

```bash
# Bhojpuri
python scripts/translate_hindi_to_lowres_openai.py --target-lang bho

# Magahi
python scripts/translate_hindi_to_lowres_openai.py --target-lang mag

# Maithili
python scripts/translate_hindi_to_lowres_openai.py --target-lang mai
```

### All Languages at Once

```bash
#!/bin/bash
for lang in bho mag mai; do
    echo "Translating to $lang..."
    python scripts/translate_hindi_to_lowres_openai.py --target-lang $lang
    echo "Done with $lang!"
    echo "---"
done
```

## How Prompt Caching Works

### System Prompt (Cached)

Contains:
- Translation guidelines
- Linguistic features of target language
- **50 monolingual example sentences** (from PV-BMM-Public-Data)

**Size**: ~5,000 tokens

### Cost Breakdown

**Without Caching** (66,317 sentences):
```
System prompt: 5,000 tokens × 66,317 = 331M tokens
Cost: 331M × $0.10/1M = $33.10
```

**With Caching** (66,317 sentences):
```
First call: 5,000 tokens × $0.10/1M = $0.0005
Remaining:  5,000 tokens × 66,316 × $0.05/1M = $16.58
Total system cost: $16.58 (50% savings!)
```

Plus translation text costs (input + output): ~$2/language

**Total per language: ~$2-3 USD**

## Output Structure

Input:
```
input/converted/Hindi/plain-text/by_file/
└── AGRICULTURE/
    └── mor-hin_agriculture_set6-pos-chunk-1-50-posn-name.txt
```

Output:
```
output/Bhojpuri/plain-text/by_file/
└── AGRICULTURE/
    └── mor-hin_agriculture_set6-pos-chunk-1-50-posn-name.txt
```

**Same directory structure, same filenames, translated content**

## Monitoring Progress

The script shows real-time progress:

```
[15/1432] AGRICULTURE/file_015.txt
  Sentences: 51
  Translating batch 1-10/51... Done (cost: $0.0018)
  Translating batch 11-20/51... Done (cost: $0.0012)
  Translating batch 21-30/51... Done (cost: $0.0012)
  Translating batch 31-40/51... Done (cost: $0.0012)
  Translating batch 41-51/51... Done (cost: $0.0013)
  ✓ Written: output/Bhojpuri/.../file_015.txt
```

Final summary:
```
TRANSLATION SUMMARY
======================================================================
Files processed:       1432
Files skipped:         0
Sentences translated:  66,317
Total cost:            $2.45 USD
Average cost/sentence: $0.000037 USD
======================================================================
```

## Customization

### Use Different Model

```bash
# Use GPT-4o-mini instead (slightly more expensive)
python scripts/translate_hindi_to_lowres_openai.py \
  --target-lang bho \
  --model gpt-4o-mini
```

### Custom Batch Size

```bash
# Translate 20 sentences per API call instead of 10
python scripts/translate_hindi_to_lowres_openai.py \
  --target-lang bho \
  --batch-size 20
```

### Custom Input/Output Directories

```bash
# Translate domain files instead
python scripts/translate_hindi_to_lowres_openai.py \
  --target-lang bho \
  --input-dir input/converted/Hindi/plain-text/by_domain \
  --output-dir output/Bhojpuri/plain-text/by_domain
```

## Verification

After translation, verify alignment using the mappings:

```bash
# Compare sentence counts
wc -l input/converted/Hindi/plain-text/by_file/AGRICULTURE/*.txt
wc -l output/Bhojpuri/plain-text/by_file/AGRICULTURE/*.txt
```

Should show identical counts for each file.

## Troubleshooting

### API Key Error
```
Error: OpenAI API key not provided
```

**Solution**:
```bash
export OPENAI_API_KEY="sk-..."
```

### Rate Limit Error
```
Error: Rate limit exceeded
```

**Solutions**:
1. Reduce batch size: `--batch-size 5`
2. Add delay between files (modify script)
3. Upgrade OpenAI plan

### Out of Memory
```
Error: MemoryError
```

**Solution**: Reduce batch size:
```bash
python scripts/translate_hindi_to_lowres_openai.py \
  --target-lang bho \
  --batch-size 5
```

## Expected Timeline

On a good internet connection:
- **5 files**: ~2-3 minutes
- **100 files**: ~30-40 minutes
- **All 1,432 files**: ~8-10 hours

## Cost Summary

| Language | Files | Sentences | Est. Cost | Actual May Vary |
|----------|-------|-----------|-----------|-----------------|
| Bhojpuri | 1,432 | 66,317    | ~$2.50    | ±$0.50          |
| Magahi   | 1,432 | 66,317    | ~$2.50    | ±$0.50          |
| Maithili | 1,432 | 66,317    | ~$2.50    | ±$0.50          |
| **Total**| 4,296 | 198,951   | **~$7.50**| ±$1.50          |

**Breakdown**:
- Input tokens (with caching): ~$1.50/lang
- Output tokens: ~$1.00/lang
- **Total per language: ~$2.50**

## What's Special

1. **Monolingual Examples**: Uses actual Bhojpuri/Magahi/Maithili sentences for context
2. **Prompt Caching**: 50% cost savings on repeated system prompts
3. **GPT-5-nano**: Latest nano model, cost-effective
4. **Structure Preservation**: Exact directory/file structure maintained
5. **Sentence Alignment**: One-to-one mapping for verification

## Next Steps

After translation:
1. Verify output file counts match input
2. Spot-check translation quality on samples
3. Use with JSON mappings for parallel corpus creation
4. Train NLP models on the parallel data

## Files Created

- `universal_translate/providers/openai_provider.py` - GPT-5-nano integration
- `universal_translate/config/prompts/hi_to_*_openai.yaml` - Prompt configs
- `universal_translate/data/examples/hi_to_*_monolingual.json` - Examples
- `scripts/translate_hindi_to_lowres_openai.py` - Translation script
- `scripts/prepare_monolingual_examples.py` - Example preparation

## Documentation

Full details:
- `OPENAI_TRANSLATION_USAGE.md` - Comprehensive usage guide
- `universal_translate/README.md` - Package documentation
- `TRANSLATION_PACKAGE_DESIGN.md` - Architecture details

---

**Ready to translate? Start with a test run:**

```bash
python scripts/translate_hindi_to_lowres_openai.py --target-lang bho --max-files 5
```
