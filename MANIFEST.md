# Project Transfer Manifest

**BhaashikLowsResLangTrans v1.0.0**

This document lists all files required to recreate the project on another computer.

## Essential Files Created for Transfer

### Environment & Configuration
- ✅ `environment.yml` - Conda environment specification (Python 3.12, all dependencies)
- ✅ `pyproject.toml` - Python package configuration (pip-installable)
- ✅ `requirements.txt` - Pip dependencies list
- ✅ `.env.example` - Environment variables template (API keys, paths)
- ✅ `.gitignore` - Git ignore rules

### Documentation
- ✅ `SETUP.md` - Complete setup guide (prerequisites, installation, verification)
- ✅ `TRANSFER_CHECKLIST.md` - Step-by-step transfer guide
- ✅ `FULL_TRANSLATION_GUIDE.md` - Complete translation usage guide
- ✅ `QUICK_START_OPENAI.md` - Quick reference guide
- ✅ `GPT-4O-MINI-FIXED.md` - GPT-4o-mini implementation details
- ✅ `MANIFEST.md` - This file (transfer manifest)
- ✅ `README.md` - Updated with setup instructions

### Core Package (`universal_translate/`)
- ✅ `core/` - Base classes (BaseTranslator, TranslationRequest, etc.)
- ✅ `providers/` - Translation providers (OpenAI, Anthropic)
- ✅ `prompts/` - Prompt management system
- ✅ `config/prompts/` - Language-specific YAML configurations
- ✅ `data/examples/` - Monolingual examples for prompt caching

### Scripts
- ✅ `scripts/translate_hindi_to_lowres_openai.py` - Main translation script
- ✅ `scripts/prepare_monolingual_examples.py` - Example preparation
- ✅ `scripts/extract_plaintext_from_conllu.py` - CoNLL-U extraction

## File Checksums

Generate checksums after setup:
```bash
md5sum environment.yml pyproject.toml requirements.txt > checksums.txt
find universal_translate/config -name "*.yaml" -exec md5sum {} \; >> checksums.txt
find universal_translate/data -name "*.json" -exec md5sum {} \; >> checksums.txt
```

## Transfer Size Estimates

| Component | Size | Transfer Method |
|-----------|------|-----------------|
| **Code + Config** | ~5 MB | Git clone |
| **Documentation** | ~500 KB | Git clone |
| **Input data** | ~200 MB | Separate transfer (rsync/cloud) |
| **Output data** (optional) | ~300 MB | Separate transfer |
| **Total (without data)** | ~6 MB | Git repository |
| **Total (with data)** | ~500 MB | Git + data transfer |

## Verification Commands

Run these on target computer after transfer:

```bash
# 1. Check files exist
ls -la environment.yml pyproject.toml requirements.txt .env.example

# 2. Check package structure
tree -L 2 universal_translate/

# 3. Verify environment creation
conda env create -f environment.yml --dry-run

# 4. Count input files
find input/converted/Hindi/plain-text/by_file -name "*.txt" | wc -l
# Expected: 1432

# 5. Check config files
find universal_translate/config/prompts -name "*.yaml"
# Expected: 3 files (hi_to_bho_openai.yaml, hi_to_mag_openai.yaml, hi_to_mai_openai.yaml)

# 6. Check examples
find universal_translate/data/examples -name "*.json"
# Expected: 4 files (3 monolingual + 1 parallel)
```

## Quick Transfer Guide

### Method 1: Git Clone (Recommended)
```bash
# On target computer
git clone https://github.com/bhaashik/BhaashikLowsResLangTrans.git
cd BhaashikLowsResLangTrans
conda env create -f environment.yml
conda activate bhaashik-translation
pip install -e .
cp .env.example .env
# Edit .env with your API key
```

### Method 2: Archive Transfer
```bash
# On source computer
tar -czf bhaashik_transfer.tar.gz \
  --exclude='.git' \
  --exclude='output' \
  --exclude='__pycache__' \
  BhaashikLowsResLangTrans/

# Transfer archive to target computer

# On target computer
tar -xzf bhaashik_transfer.tar.gz
cd BhaashikLowsResLangTrans
# Follow setup from environment.yml
```

## Dependencies Summary

### Python Packages
- openai >= 2.8.0
- conllu >= 6.0.0
- pyyaml >= 6.0
- python-dotenv >= 1.0.0
- requests >= 2.31.0
- tqdm >= 4.65.0

### System Requirements
- Python 3.10+ (3.12 recommended)
- Conda/Miniconda
- 50+ GB disk space
- Internet connection (for API calls)

## Post-Transfer Testing

Minimal test to verify everything works:
```bash
# Activate environment
conda activate bhaashik-translation

# Test imports
python -c "from universal_translate import TranslationRequest; print('✓ Package OK')"

# Test API key
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('✓ API key:', 'SET' if os.getenv('OPENAI_API_KEY') else 'NOT SET')"

# Test translation (1 file, ~$0.002)
python scripts/translate_hindi_to_lowres_openai.py --target-lang bho --max-files 1

# Verify output
head -3 output/Bhojpuri/plain-text/by_file/AGRICULTURE/*.txt
```

Expected: Non-empty Bhojpuri translations

## Support Documentation

All documentation is included in the repository:

1. **Setup:** `SETUP.md`
2. **Transfer:** `TRANSFER_CHECKLIST.md`
3. **Usage:** `FULL_TRANSLATION_GUIDE.md`, `QUICK_START_OPENAI.md`
4. **Architecture:** `CLAUDE.md`
5. **Technical Details:** `GPT-4O-MINI-FIXED.md`

## Version Information

- **Project Version:** 1.0.0
- **Python Version:** 3.12.x (compatible with 3.10+)
- **Last Updated:** 2025-12-04
- **Tested On:** Ubuntu 24.04, WSL2, macOS

## Notes

- `.env` file contains API keys - transfer securely and separately
- `output/` directory can be large - transfer only if needed
- Git repository does not include large data files
- Input data should be transferred separately via rsync/cloud
