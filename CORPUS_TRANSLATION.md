# Corpus Translation Guide

Complete guide for translating entire corpus structures with directory preservation.

## Overview

The `translate_corpus.py` script enables translation of complete corpus structures while preserving:
- Directory hierarchy
- File organization
- Metadata (in CoNLL-U files)
- Sentence boundaries

**Perfect for:**
- Universal Dependencies treebanks
- Parallel corpus creation
- Multi-domain datasets
- Structured text collections

## Quick Start

### Translate Plain Text Corpus

```bash
python scripts/translate_corpus.py \
  --input-dir corpus/en \
  --output-dir corpus/hi \
  --src en --tgt hi \
  --format txt \
  --indictrans2
```

### Translate Universal Dependencies (CoNLL-U)

```bash
python scripts/translate_corpus.py \
  --input-dir UD_Hindi-HDTB \
  --output-dir UD_Bhojpuri-HDTB \
  --src hi --tgt bho \
  --format conllu \
  --hindi-pivot
```

## Supported Formats

### 1. Plain Text (.txt)

**Input format:** One sentence per line

```
Hello, how are you?
What is your name?
I am learning Hindi.
```

**Output format:** Same structure, translated

```
नमस्ते, आप कैसे हैं?
आपका नाम क्या है?
मैं हिंदी सीख रहा हूं।
```

### 2. CoNLL-U (.conllu)

**Input format:** Universal Dependencies format

```conllu
# sent_id = 1
# text = Hello world
1    Hello    hello    INTJ    _    _    0    root       _    _
2    world    world    NOUN    _    _    1    vocative   _    _
```

**Output format:** Preserves structure, translates text

```conllu
# sent_id = 1
# text_en = Hello world
# text = नमस्ते दुनिया
1    Hello    hello    INTJ    _    _    0    root       _    _
2    world    world    NOUN    _    _    1    vocative   _    _
```

**Key features:**
- All token annotations preserved (POS, dependencies, features)
- Original text stored as `# text_<src_lang> = ...`
- Translation stored as `# text = ...`
- Sentence IDs and metadata preserved

## Translation Strategies

### 1. IndicTrans2 (FREE - Recommended for 22 languages)

```bash
python scripts/translate_corpus.py \
  --input-dir input/ \
  --output-dir output/ \
  --src en --tgt hi \
  --format txt \
  --indictrans2
```

**Best for:**
- Assamese, Bengali, Gujarati, Hindi, Kannada, and other scheduled languages
- High quality translations
- Zero cost (compute only)

### 2. Hindi Pivot (FREE - For low-resource languages)

```bash
python scripts/translate_corpus.py \
  --input-dir input/ \
  --output-dir output/ \
  --src en --tgt bho \
  --format txt \
  --hindi-pivot
```

**Best for:**
- Bhojpuri, Magahi, Awadhi, Braj, Marwari, Bundeli
- Free translation using Hindi as bridge
- Linguistically close languages

### 3. NLLB (FREE - Alternative)

```bash
python scripts/translate_corpus.py \
  --input-dir input/ \
  --output-dir output/ \
  --src en --tgt hi \
  --format txt \
  --nllb
```

**Best for:**
- Broader language coverage
- Alternative to IndicTrans2
- Good quality

### 4. Tiered Strategy (COSTS MONEY - Best quality)

```bash
python scripts/translate_corpus.py \
  --input-dir input/ \
  --output-dir output/ \
  --src en --tgt bho \
  --format txt \
  --tiered \
  --tier-free 70 \
  --tier-enhancement 20 \
  --tier-premium 10
```

**Best for:**
- High-quality translation needs
- Low-resource languages
- Production use cases

**Cost example:** ~₹5,000 per 1M sentences with 70/20/10 mix

## Directory Structure Preservation

### Input Structure

```
corpus/
├── train/
│   ├── news.txt
│   ├── social.txt
│   └── formal/
│       └── documents.txt
├── dev/
│   └── validation.txt
└── test/
    └── evaluation.txt
```

### Output Structure (Automatically Created)

```
translated_corpus/
├── train/
│   ├── news.txt              # Translated
│   ├── social.txt            # Translated
│   └── formal/
│       └── documents.txt     # Translated
├── dev/
│   └── validation.txt        # Translated
├── test/
│   └── evaluation.txt        # Translated
└── translation_metadata.json # Statistics
```

## Advanced Options

### Pattern-Based File Selection

Translate only specific files:

```bash
# Only train files
python scripts/translate_corpus.py \
  --input-dir UD_Hindi \
  --output-dir UD_Bhojpuri \
  --src hi --tgt bho \
  --format conllu \
  --pattern 'train/*.conllu' \
  --indictrans2

# Only .txt files in specific directory
python scripts/translate_corpus.py \
  --input-dir corpus \
  --output-dir translated \
  --src en --tgt hi \
  --format txt \
  --pattern 'news/**/*.txt' \
  --indictrans2
```

### Resume Capability

By default, the script skips already translated files:

```bash
# First run (interrupted)
python scripts/translate_corpus.py --input-dir corpus --output-dir output ...

# Resume automatically (skips existing files)
python scripts/translate_corpus.py --input-dir corpus --output-dir output ...

# Force retranslation
python scripts/translate_corpus.py --input-dir corpus --output-dir output ... --no-resume
```

### Batch Size Optimization

Adjust batch size based on your hardware:

```bash
# GPU with large memory
python scripts/translate_corpus.py ... --batch-size 64

# CPU or limited GPU
python scripts/translate_corpus.py ... --batch-size 16

# API translation
python scripts/translate_corpus.py ... --batch-size 10
```

## Translation Metadata

The script creates `translation_metadata.json` with complete statistics:

```json
{
  "source_dir": "/path/to/input",
  "output_dir": "/path/to/output",
  "src_lang": "en",
  "tgt_lang": "hi",
  "translator_type": "indictrans2",
  "file_format": "txt",
  "timestamp": "2025-01-15T10:30:00",
  "total_files": 50,
  "successful_files": 48,
  "failed_files": 2,
  "total_sentences": 125000,
  "files": [
    {
      "input": "train/data.txt",
      "output": "train/data.txt",
      "sentences": 5000,
      "status": "success"
    },
    ...
  ]
}
```

## Real-World Examples

### 1. Translate Universal Dependencies Treebank

```bash
# Download UD Hindi treebank first
# https://universaldependencies.org/

# Translate to Bhojpuri
python scripts/translate_corpus.py \
  --input-dir ~/corpora/UD_Hindi-HDTB \
  --output-dir ~/corpora/UD_Bhojpuri-HDTB \
  --src hi --tgt bho \
  --format conllu \
  --tiered \
  --batch-size 32

# Result: Complete UD treebank in Bhojpuri!
```

### 2. Create Multi-Source Parallel Corpus

```bash
# Translate from multiple source languages
for src in hi bn mr ta; do
  python scripts/translate_corpus.py \
    --input-dir corpora/UD_${src} \
    --output-dir corpora/UD_Bhojpuri_from_${src} \
    --src ${src} --tgt bho \
    --format conllu \
    --hindi-pivot
done

# Combine results for better coverage
```

### 3. Domain-Specific Corpus Translation

```bash
# Translate multi-domain corpus
# Structure: corpus/{news,social,formal,technical}/files.txt

python scripts/translate_corpus.py \
  --input-dir corpus_en \
  --output-dir corpus_hi \
  --src en --tgt hi \
  --format txt \
  --indictrans2 \
  --batch-size 48

# Preserves domain organization automatically
```

### 4. Large-Scale Parallel Corpus Creation

```bash
# For very large corpora
python scripts/translate_corpus.py \
  --input-dir large_corpus/en \
  --output-dir large_corpus/bho \
  --src en --tgt bho \
  --format txt \
  --tiered \
  --tier-free 85 \
  --tier-enhancement 10 \
  --tier-premium 5 \
  --batch-size 64

# Monitor with:
# tail -f logs/translation.log
```

## Performance Tips

### 1. Optimize Batch Size

| Hardware | Format | Recommended Batch Size |
|----------|--------|----------------------|
| GPU (24GB) | txt | 64-128 |
| GPU (12GB) | txt | 32-64 |
| GPU (8GB) | txt | 16-32 |
| CPU | txt | 8-16 |
| API | Any | 10-20 |
| CoNLL-U | Any | Half of txt batch size |

### 2. Resume for Long Jobs

```bash
# Use screen or tmux for long-running jobs
screen -S translation

python scripts/translate_corpus.py ...

# Detach: Ctrl+A, D
# Reattach: screen -r translation
```

### 3. Monitor Progress

```bash
# Terminal 1: Run translation
python scripts/translate_corpus.py ...

# Terminal 2: Monitor logs
tail -f logs/translation.log

# Terminal 3: Check output
watch -n 10 'find output_dir -type f | wc -l'
```

## Troubleshooting

### Issue: Out of Memory

**Solution:** Reduce batch size

```bash
python scripts/translate_corpus.py ... --batch-size 8
```

### Issue: Translation too slow

**Solutions:**
1. Increase batch size (if memory allows)
2. Use GPU instead of CPU
3. Use distilled models (200M instead of 1B)

### Issue: Poor quality translations

**Solutions:**
1. Switch from hindi-pivot to tiered strategy
2. Increase API percentage in tiered mix
3. Try different source language (if available)

### Issue: Files not found

**Solution:** Check pattern syntax

```bash
# Correct patterns:
--pattern '**/*.txt'      # All .txt files recursively
--pattern 'train/*.txt'   # Only in train/ directory
--pattern '*.conllu'      # Only in root directory

# Check what files match:
ls input_dir/**/*.txt
```

## Cost Estimation

Before translating large corpora with paid APIs:

```bash
# Count sentences first
find input_dir -name "*.txt" -exec wc -l {} + | tail -1

# Estimate cost using translate.py
python scripts/translate.py \
  --src en --tgt bho \
  --num-samples <sentence_count> \
  --tier-free 70 \
  --tier-enhancement 20 \
  --tier-premium 10 \
  --estimate-only
```

## Best Practices

1. **Start Small**
   - Test on a subset first (use --pattern)
   - Verify quality before full translation

2. **Choose Right Strategy**
   - FREE methods first (IndicTrans2, Hindi Pivot)
   - Tiered only if quality insufficient

3. **Monitor Carefully**
   - Check translation_metadata.json
   - Review sample outputs
   - Monitor costs for API usage

4. **Preserve Originals**
   - Keep original corpus untouched
   - Use separate output directory
   - Version control your translations

5. **Quality Check**
   - Sample random files for manual review
   - Use quality metrics on dev set
   - Compare different translation strategies

## Integration with Workflows

### Post-Processing Pipeline

```bash
# 1. Translate corpus
python scripts/translate_corpus.py ...

# 2. Validate output
python scripts/verify.py --check-corpus output_dir

# 3. Calculate quality metrics
python scripts/evaluate_quality.py \
  --predictions output_dir \
  --references reference_dir

# 4. Post-edit if needed
python scripts/post_edit.py --input output_dir
```

### Continuous Translation

```bash
#!/bin/bash
# Translate new files as they arrive

inotifywait -m input_dir -e create -e moved_to |
while read path action file; do
    python scripts/translate_corpus.py \
      --input-dir input_dir \
      --output-dir output_dir \
      --src en --tgt hi \
      --format txt \
      --indictrans2
done
```

## See Also

- [USAGE.md](USAGE.md) - General usage guide
- [Example 5](examples/example_5_corpus_translation.py) - Working examples
- [Universal Dependencies](https://universaldependencies.org/) - UD treebanks
- [CoNLL-U Format](https://universaldependencies.org/format.html) - Format specification
