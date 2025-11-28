# Plain Text Extraction from CoNLL-U Files

## Overview

This directory contains plain text extracted from CoNLL-U format files, maintaining the exact same directory structure and sentence order as the original files.

## Extraction Details

- **Date**: December 1, 2025
- **Script**: `scripts/extract_plaintext_from_conllu.py`
- **Library**: `conllu` (v6.0.0)
- **Total Sentences**: 116,550
- **Total Files**: 1,446

## Directory Structure

```
Hindi/
├── plain-text/              # Extracted plain text files
│   ├── by_domain/          # Domain-aggregated text files (11 domains)
│   │   ├── Hindi_AGRICULTURE.txt
│   │   ├── Hindi_BOX-OFFICE.txt
│   │   ├── Hindi_CONVERSATIONAL.txt
│   │   ├── Hindi_CRICKET.txt
│   │   ├── Hindi_DISEASE.txt
│   │   ├── Hindi_ENTERTAINMENT.txt
│   │   ├── Hindi_GADGET.txt
│   │   ├── Hindi_JUDICIARY.txt
│   │   ├── Hindi_NEWS-ARTICLES.txt
│   │   ├── Hindi_RECIPE.txt
│   │   └── Hindi_TOURISM.txt
│   │
│   ├── by_file/            # Individual file-level text (1,435 files)
│   │   ├── AGRICULTURE/
│   │   ├── BOX-OFFICE/
│   │   ├── CONVERSATIONAL/
│   │   ├── CRICKET/
│   │   ├── DISEASE/
│   │   ├── ENTERTAINMENT/
│   │   ├── GADGET/
│   │   ├── JUDICIARY/
│   │   ├── NEWS-ARTICLES/
│   │   ├── RECIPE/
│   │   └── TOURISM/
│   │
│   └── merged/             # Universal Dependencies merged files (3 files)
│       ├── hi_hdtb-ud-dev.txt    (5,001 sentences)
│       ├── hi_hdtb-ud-test.txt   (5,027 sentences)
│       └── hi_hdtb-ud-train.txt  (40,043 sentences)
│
└── mappings/               # JSON mapping files (sent_id → sentence position)
    ├── hindi_by_domain_mappings.json
    ├── hindi_by_file_mappings.json
    ├── hindi_merged_mappings.json
    └── hindi_extraction_stats.json
```

## Sentence Statistics by Domain

| Domain | Sentences |
|--------|-----------|
| NEWS-ARTICLES | 18,863 |
| TOURISM | 3,022 |
| JUDICIARY | 2,257 |
| ENTERTAINMENT | 2,178 |
| CONVERSATIONAL | 2,022 |
| DISEASE | 1,824 |
| AGRICULTURE | 1,012 |
| CRICKET | 555 |
| GADGET | 564 |
| RECIPE | 549 |
| BOX-OFFICE | 517 |

## Mapping Files

### 1. By Domain Mappings (`hindi_by_domain_mappings.json`)

Maps CoNLL-U `sent_id` to 1-based sentence position within each domain file.

**Format**:
```json
{
  "Hindi_AGRICULTURE": {
    "AGRICULTURE-mor-hin_agriculture_set6-pos-chunk-1-50-posn-name-s1": 1,
    "AGRICULTURE-mor-hin_agriculture_set6-pos-chunk-1-50-posn-name-s2": 2,
    ...
  },
  "Hindi_BOX-OFFICE": { ... },
  ...
}
```

### 2. By File Mappings (`hindi_by_file_mappings.json`)

Maps CoNLL-U `sent_id` to 1-based sentence position within each individual file.

**Format**:
```json
{
  "AGRICULTURE": {
    "mor-hin_agriculture_set6-pos-chunk-1-50-posn-name": {
      "mor-hin_agriculture_set6-pos-chunk-1-50-posn-name-s1": 1,
      "mor-hin_agriculture_set6-pos-chunk-1-50-posn-name-s2": 2,
      ...
    },
    ...
  },
  "BOX-OFFICE": { ... },
  ...
}
```

### 3. Merged Mappings (`hindi_merged_mappings.json`)

Maps CoNLL-U `sent_id` to 1-based sentence position in UD merged files.

**Format**:
```json
{
  "hi_hdtb-ud-dev": {
    "dev-s1": 1,
    "dev-s2": 2,
    ...
  },
  "hi_hdtb-ud-test": { ... },
  "hi_hdtb-ud-train": { ... }
}
```

### 4. Extraction Statistics (`hindi_extraction_stats.json`)

Contains summary statistics for the extraction process.

## Extraction Methodology

1. **Text Reconstruction**:
   - Primary: Uses `# text = ...` metadata from CoNLL-U files when available
   - Fallback: Reconstructs text by joining token forms (skips multiword tokens)

2. **Order Preservation**:
   - Sentences are extracted in the exact order they appear in CoNLL-U files
   - No reordering or shuffling applied

3. **Mapping Creation**:
   - Each sentence gets a 1-based index position
   - `sent_id` from CoNLL-U metadata maps to this position
   - Enables reverse lookup: position → original CoNLL-U sentence

## Use Cases

### 1. Translation Verification

After translating plain text files, use mappings to verify against original CoNLL-U:

```python
import json

# Load mapping
with open('mappings/hindi_by_domain_mappings.json') as f:
    mapping = json.load(f)

# Find original sentence by sent_id
sent_id = "AGRICULTURE-mor-hin_agriculture_set6-pos-chunk-1-50-posn-name-s5"
position = mapping['Hindi_AGRICULTURE'][sent_id]  # Returns: 5

# Read corresponding translated sentence
with open('output/Bhojpuri/plain-text/by_domain/Bhojpuri_AGRICULTURE.txt') as f:
    lines = f.readlines()
    translated_sentence = lines[position - 1]  # 1-based to 0-based
```

### 2. Quality Assessment

Compare source and translation at sentence level:

```python
# Get sentence at position N
position = 100
domain = "Hindi_AGRICULTURE"

# Find sent_id for this position
sent_id = [k for k, v in mapping[domain].items() if v == position][0]

# Now can look up original CoNLL-U annotations for quality checks
```

### 3. Batch Translation

The plain text files can be directly fed to translation systems:

```bash
# Example: Translate to Bhojpuri using Hindi pivot
python scripts/translate_corpus.py \
  --input-dir input/converted/Hindi/plain-text/by_domain \
  --output-dir output/Bhojpuri/plain-text/by_domain \
  --src hi --tgt bho --format txt --hindi-pivot
```

## File Naming Convention

- **Original CoNLL-U**: `[filename].conllu`
- **Extracted Text**: `[filename].txt`
- **Exact same basename** to maintain 1:1 correspondence

## Verification

To verify extraction integrity:

```bash
# Check sentence counts match
wc -l input/converted/Hindi/plain-text/merged/*.txt
# Should show: 5001, 5027, 40043 sentences

# Verify mappings exist
ls -lh input/converted/Hindi/mappings/
# Should show 4 JSON files

# Check extraction summary
cat input/converted/extraction_summary.json
```

## Notes

- **NULL tokens**: Some sentences contain `NULL` tokens (from CoNLL-U), preserved in plain text
- **Punctuation**: All original punctuation is preserved
- **Encoding**: All files use UTF-8 encoding
- **Line endings**: Unix-style (LF) line endings

## Next Steps

1. ✅ Plain text extraction complete
2. ⏳ Translation to low-resource languages (Bhojpuri, Magahi, etc.)
3. ⏳ Output placement in identical directory structure
4. ⏳ Translation verification using mappings

## Script Usage

To re-run extraction:

```bash
python scripts/extract_plaintext_from_conllu.py
```

The script will:
- Process all `.conllu` files in `input/converted/[Language]/`
- Create `plain-text/` subdirectories
- Generate mapping files in `mappings/`
- Produce extraction statistics
