# Package Manifest - GPT-5 Nano Hindi Translation

**Package Name**: GPT-5 Nano Hindi Translation Toolkit  
**Version**: 1.0  
**Date**: November 2025  
**Total Files**: 7

---

## File List

### 1. Core Scripts (2 files)

| File | Size | Purpose |
|------|------|---------|
| `translate_domain_corpus.py` | ~30 KB | Main translation script |
| `setup_wsl2.sh` | ~8 KB | Automated WSL2 setup |

### 2. Documentation (4 files)

| File | Size | Purpose |
|------|------|---------|
| `README.md` | ~20 KB | Complete documentation |
| `QUICKSTART.md` | ~2 KB | 2-minute quick start |
| `TROUBLESHOOTING.md` | ~12 KB | Issue resolution guide |
| `MANIFEST.md` | ~2 KB | This file |

### 3. Configuration (1 file)

| File | Size | Purpose |
|------|------|---------|
| `requirements.txt` | <1 KB | Python dependencies |

### 4. Utilities (1 file)

| File | Size | Purpose |
|------|------|---------|
| `create_sample_corpus.sh` | ~4 KB | Generate test corpus |

---

## Total Package Size

- **Scripts & Docs**: ~78 KB
- **Compressed**: ~25 KB (if zipped)

---

## File Descriptions

### translate_domain_corpus.py
Main Python script for translating Hindi to Bhojpuri and Magahi.

**Features**:
- Domain-organized corpus support
- Automatic temp file exclusion (.swp, .tmp, .bak)
- Batch processing (50 sentences/call)
- Progress tracking
- Cost estimation
- Error handling
- Directory structure preservation

**Usage**:
```bash
python translate_domain_corpus.py
```

---

### setup_wsl2.sh
Automated setup script for WSL2 Ubuntu environments.

**What it does**:
- Checks Conda installation
- Creates/activates environment
- Installs dependencies
- Sets up API key
- Creates project directory
- Generates example corpus

**Usage**:
```bash
bash setup_wsl2.sh
```

---

### README.md
Complete documentation covering:
- Installation instructions
- Usage guide
- Cost information
- Features list
- Troubleshooting
- Examples
- FAQ

**Start here** for comprehensive information.

---

### QUICKSTART.md
Ultra-condensed quick start guide for users who want to get running in 2 minutes.

**Perfect for**: Experienced users who just need the commands.

---

### TROUBLESHOOTING.md
Dedicated troubleshooting guide covering:
- Installation issues
- API key problems
- File/directory errors
- WSL2-specific issues
- Translation problems
- Cost/performance issues
- Diagnostic commands

**Reference when**: Things don't work as expected.

---

### requirements.txt
Python package dependencies.

**Contents**:
```
openai>=1.50.0
```

**Usage**:
```bash
pip install -r requirements.txt
```

---

### create_sample_corpus.sh
Utility script to generate a sample Hindi corpus for testing.

**Creates**:
- 4 domains (news, literature, social, education)
- 8 files
- ~40 Hindi sentences
- Complete directory structure

**Usage**:
```bash
bash create_sample_corpus.sh
```

---

## Installation Order

### Recommended (WSL2 Ubuntu):
1. Run `setup_wsl2.sh` (does everything automatically)
2. Test with `create_sample_corpus.sh`
3. Run `translate_domain_corpus.py`

### Manual (any Linux):
1. Read `README.md`
2. Install from `requirements.txt`
3. Set API key (see README)
4. Run `translate_domain_corpus.py`

---

## Quick Reference

### Essential Commands:

```bash
# Setup
bash setup_wsl2.sh

# Install dependencies
pip install -r requirements.txt

# Set API key
export OPENAI_API_KEY="sk-proj-..."

# Create test corpus
bash create_sample_corpus.sh

# Run translation
python translate_domain_corpus.py
```

---

## System Requirements

- **OS**: Ubuntu 20.04+ (WSL2 or native), or any Linux
- **Python**: 3.8+
- **Conda**: Optional (recommended)
- **Disk**: 10+ GB free
- **Network**: Stable internet connection
- **API**: OpenAI API key with credits

---

## Dependencies

### Python Packages:
- `openai` - OpenAI Python SDK (only required package)

### System Tools (optional):
- `conda` - Environment management
- `tree` - Directory visualization
- `bc` - Calculator (for cost estimates)

---

## Features Summary

✅ Domain-aware translation  
✅ Directory structure preservation  
✅ Automatic temp file exclusion  
✅ Batch processing for efficiency  
✅ Real-time progress tracking  
✅ Comprehensive error handling  
✅ Cost estimation before running  
✅ Translation statistics  
✅ Linguistic expertise built-in  
✅ GPT-5 Nano optimized (ultra-fast, ultra-cheap)  

---

## Language Support

**Currently Supported**:
- Hindi → Bhojpuri (भोजपुरी)
- Hindi → Magahi (मगही)

**Extensible**:
- Can add more languages by editing SYSTEM_PROMPTS
- Maithili available FREE via IndicTrans2 (see docs)

---

## Cost Estimate

Using GPT-5 Nano:

| Sentences | Cost per Language | Total (2 langs) |
|-----------|-------------------|-----------------|
| 1,000 | ₹3 ($0.03) | ₹6 ($0.06) |
| 10,000 | ₹26 ($0.30) | ₹52 ($0.60) |
| 35,000 | ₹91 ($1.07) | ₹182 ($2.14) |
| 70,000 | ₹183 ($2.15) | ₹366 ($4.30) |
| 100,000 | ₹261 ($3.07) | ₹522 ($6.14) |

---

## Version History

### Version 1.0 (November 2025)
- Initial release
- GPT-5 Nano support
- Bhojpuri & Magahi translation
- Domain-organized corpus support
- WSL2 setup automation
- Comprehensive documentation

---

## File Integrity

To verify files after download:

```bash
# Check if all files present
ls -1 *.py *.sh *.md *.txt

# Should show:
# translate_domain_corpus.py
# setup_wsl2.sh
# README.md
# QUICKSTART.md
# TROUBLESHOOTING.md
# MANIFEST.md
# requirements.txt
# create_sample_corpus.sh
```

---

## Support

For issues:
1. Check TROUBLESHOOTING.md
2. Review README.md
3. Verify API key and credits
4. Test with sample corpus
5. Check OpenAI status: https://status.openai.com

---

## License

Scripts provided for research and educational purposes.

**API Usage**: Subject to OpenAI terms of service.

---

## Changelog

**1.0** (November 2025):
- Initial public release
- Complete translation toolkit
- Full documentation suite
- Automated setup for WSL2
- Sample corpus generator

---

**Package Ready**: Yes ✅  
**Documentation**: Complete ✅  
**Tested**: WSL2 Ubuntu 24.04 ✅  
**Production Ready**: Yes ✅  

---

**Download all files and start translating!** 🚀
