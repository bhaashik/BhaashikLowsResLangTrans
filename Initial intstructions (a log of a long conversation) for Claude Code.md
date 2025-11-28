# Synthetic Parallel Corpus Setup for Low-Resource Indic Languages

**Project Date:** November 2025  
**Environment:** Ubuntu 24.04 with Conda  
**Focus:** Creating synthetic parallel corpora for low-resource Indic and Dravidian languages

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Translation Options Analysis](#translation-options-analysis)
3. [Budget Analysis](#budget-analysis)
4. [Complete Ubuntu Setup Guide](#complete-ubuntu-setup-guide)
5. [Download Scripts](#download-scripts)
6. [Implementation Recommendations](#implementation-recommendations)
7. [References](#references)

---

## Project Overview

### Objective
Generate synthetic parallel corpora for low-resource Indic languages, particularly those not supported by IndicTrans2's 22 scheduled languages.

### Samanantar Corpus Statistics
- **Total pairs**: 49.7 million sentence pairs
- **Languages**: 11 Indic languages (English-paired)
- **Total characters**: ~5 billion characters (including both English and Indic)
- **Disk space**: ~25-30 GB compressed, ~50-60 GB uncompressed

### Target Languages

#### Supported by IndicTrans2 (FREE - 22 scheduled languages):
- Assamese, Bengali, Gujarati, Hindi, Kannada, Kashmiri, Konkani, Malayalam, Manipuri, Marathi, Nepali, Odia, Punjabi, Sanskrit, Sindhi, Tamil, Telugu, Urdu, Bodo, Santhali, Maithili, Dogri

#### NOT Supported (Need Budget - 6 languages):
1. **Bhojpuri** (Indo-Aryan, ~51M speakers)
2. **Magahi** (Indo-Aryan, ~13M speakers)
3. **Awadhi** (Indo-Aryan, ~38M speakers)
4. **Braj** (Indo-Aryan, ~1M speakers)
5. **Marwari** (Indo-Aryan, ~13M speakers)
6. **Bundeli** (Indo-Aryan, ~3M speakers)

---

## Translation Options Analysis

### 1. Commercial APIs

| Service | Cost/1M chars | Coverage | Notes |
|---------|--------------|----------|-------|
| **Google Cloud Translation** | $20 | Good for scheduled languages | High quality |
| **Microsoft Azure Translator** | $10 | Good for major Indic | 2M free/month |
| **OpenAI GPT-4 Turbo** | $10-30 | All languages (variable) | Variable quality |
| **OpenAI GPT-3.5 Turbo** | $1-3 | All languages | Cost-effective |
| **Claude Sonnet 4.5** | $3-15 | All languages | High quality |
| **Claude Haiku 4.5** | $1-5 | All languages | Best value |
| **AWS Translate** | $15 | Limited Indic | Limited coverage |

### 2. Open Source Models

| Model | Coverage | Cost | Quality |
|-------|----------|------|---------|
| **IndicTrans2** | 22 scheduled languages | FREE (compute only) | Excellent for supported |
| **NLLB-200** | 200+ languages | FREE (compute only) | Good for most Indic |
| **Llama 3.1** | All (via prompting) | FREE (compute only) | Variable |
| **Mistral/Mixtral** | All (via prompting) | FREE (compute only) | Variable |
| **Aya-101** | 101 languages | FREE (compute only) | Good for low-resource |

---

## Budget Analysis

### For 6 Unsupported Languages (12M sentence pairs = ~1.2B characters)

#### Scenario 1: Pure Open Source (RECOMMENDED)
```
IndicTrans2 Hindi Pivot: FREE
Compute costs only: $500-1,000
Total: $500-1,000
```

#### Scenario 2: Hybrid Approach with Claude API

**Using Indian Pricing (USD to INR, $1 = ₹85):**

| Model | Input (₹/1M tokens) | Output (₹/1M tokens) |
|-------|---------------------|----------------------|
| Claude Haiku 3 | ₹21 | ₹106 |
| Claude Haiku 3.5 | ₹68 | ₹340 |
| Claude Haiku 4.5 | ₹85 | ₹425 |
| Claude Sonnet 4.5 | ₹255 | ₹1,275 |

**Tiered Approach (Optimal):**

Without optimization:
- 70% Free pivot (IndicTrans2): ₹0
- 20% Haiku 3.5 enhancement: ₹73,440
- 10% Haiku 4.5 premium: ₹45,900
- **Total: ₹1,19,340 (~$1,405)**

With Batch API (50% output discount):
- **Total: ₹69,615 (~$819)**

With Batch API + Prompt Caching:
- **Total: ₹62,912 (~$740)**

**Per-language cost: ₹10,485 (~$123) for 2M sentence pairs**

### Cost Comparison Table

| Approach | Cost (INR) | Cost (USD) | Quality |
|----------|-----------|------------|---------|
| 100% IndicTrans2 Pivot | ₹0 | $0 | Good |
| 90% Pivot + 10% Haiku 4.5 | ₹45,900 | $540 | Good+ |
| 80% Pivot + 20% Haiku 4.5 | ₹91,800 | $1,080 | Very Good |
| Tiered (70/20/10) | ₹1,19,340 | $1,405 | Optimal |
| Tiered + Batch API | ₹69,615 | $819 | Optimal |
| **Tiered + Batch + Cache** | **₹62,912** | **$740** | **Best Value** |

---

## Complete Ubuntu Setup Guide

### Prerequisites

```bash
# System requirements
Ubuntu 24.04 or later
Python 3.10+
Minimum 120 GB disk space (150+ GB recommended)
16 GB RAM minimum (32 GB recommended)
Good internet connection (50+ Mbps recommended)
```

### Part 1: Conda Installation

```bash
# Check if conda exists
which conda

# If not installed, download Miniconda
cd /tmp
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Accept license, use default location, initialize conda
source ~/.bashrc
conda --version
```

### Part 2: Conda Configuration

```bash
# Update conda
conda update -n base -c defaults conda -y

# Configure channels
conda config --add channels conda-forge
conda config --set channel_priority strict

# Enable faster solving
conda install -n base conda-libmamba-solver -y
conda config --set solver libmamba
```

### Part 3: Create Environment

```bash
# Create environment
conda create -n NLPLResourceDownload python=3.10 -y

# Activate
conda activate NLPLResourceDownload

# Verify
which python  # Should show conda env path
```

### Part 4: Install Packages

```bash
# Core packages via conda
conda install -c conda-forge -y \
    pytorch torchvision torchaudio pytorch-cuda=11.8 \
    transformers \
    datasets \
    accelerate \
    sentencepiece \
    protobuf \
    numpy \
    pandas \
    tqdm \
    requests \
    aiohttp \
    pyyaml

# Additional packages via pip
pip install --upgrade pip
pip install \
    huggingface-hub \
    hf-transfer \
    python-dotenv \
    colorama \
    rich

# Git LFS
conda install -c conda-forge git-lfs -y
git lfs install

# System tools (optional)
sudo apt-get update
sudo apt-get install -y aria2

# Enable faster downloads
export HF_HUB_ENABLE_HF_TRANSFER=1
```

**For CPU-only systems:**
```bash
conda install -c conda-forge pytorch torchvision torchaudio cpuonly -y
```

### Part 5: Directory Setup

```bash
# Define base directory (CHANGE THIS to your desired location)
export BASE_DIR="/mnt/data/nlp_resources"

# Create structure
mkdir -p "$BASE_DIR"/{datasets,models,cache,logs,scripts}

# Set cache directories
export HF_DATASETS_CACHE="$BASE_DIR/cache/datasets"
export HF_HOME="$BASE_DIR/cache/huggingface"
export TRANSFORMERS_CACHE="$BASE_DIR/cache/transformers"

# Create cache directories
mkdir -p "$HF_DATASETS_CACHE"
mkdir -p "$HF_HOME"
mkdir -p "$TRANSFORMERS_CACHE"

# Make persistent in .bashrc
cat >> ~/.bashrc << 'EOF'

# NLP Resource Download Environment Variables
export BASE_DIR="/mnt/data/nlp_resources"
export HF_DATASETS_CACHE="$BASE_DIR/cache/datasets"
export HF_HOME="$BASE_DIR/cache/huggingface"
export TRANSFORMERS_CACHE="$BASE_DIR/cache/transformers"
export HF_HUB_ENABLE_HF_TRANSFER=1
EOF

source ~/.bashrc

# Verify
echo "BASE_DIR: $BASE_DIR"
ls -la "$BASE_DIR"
```

### Part 6: HuggingFace Authentication

```bash
# Login (required for Llama and other gated models)
huggingface-cli login

# Get token from: https://huggingface.co/settings/tokens
# Paste and press Enter
```

---

## Download Scripts

### Script 1: Master Download Script

Save as `$BASE_DIR/scripts/master_download.sh`:

```bash
#!/bin/bash
#
# Master Download Script for NLP Resources
# Usage: ./master_download.sh [all|samanantar|indictrans|nllb|llms]
#

set -e  # Exit on error

# Source environment variables
source ~/.bashrc

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate NLPLResourceDownload

# Configuration
BASE_DIR="${BASE_DIR:-/mnt/data/nlp_resources}"
LOGS_DIR="$BASE_DIR/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOGS_DIR/download_${TIMESTAMP}.log"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Logging functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

# Check disk space
check_disk_space() {
    local required_gb=$1
    local available_gb=$(df -BG "$BASE_DIR" | awk 'NR==2 {print $4}' | sed 's/G//')
    
    log "Available: ${available_gb}GB, Required: ${required_gb}GB"
    
    if [ "$available_gb" -lt "$required_gb" ]; then
        error "Insufficient disk space!"
        exit 1
    fi
}

# Download with retry
download_with_retry() {
    local repo=$1
    local max_attempts=3
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        log "Attempt $attempt/$max_attempts: $repo"
        
        if huggingface-cli download "$repo" \
            --cache-dir "$HF_HOME" \
            --resume-download 2>&1 | tee -a "$LOG_FILE"; then
            log "✓ Downloaded: $repo"
            return 0
        fi
        
        warning "Retrying in 10 seconds..."
        sleep 10
        attempt=$((attempt + 1))
    done
    
    error "Failed after $max_attempts attempts: $repo"
    return 1
}

# SAMANANTAR DOWNLOAD
download_samanantar() {
    log "Starting Samanantar corpus download..."
    check_disk_space 80
    
    python3 << 'PYTHON_SCRIPT'
import sys
from datasets import load_dataset
import os

cache_dir = os.environ.get('HF_DATASETS_CACHE')
languages = ['as', 'bn', 'gu', 'hi', 'kn', 'ml', 'mr', 'or', 'pa', 'ta', 'te']

print("\n" + "="*70)
print("DOWNLOADING SAMANANTAR CORPUS")
print("="*70)

total = 0
failed = []

for lang in languages:
    print(f"\n{'='*70}")
    print(f"Language: English ↔ {lang.upper()}")
    print(f"{'='*70}")
    
    try:
        dataset = load_dataset(
            "ai4bharat/samanantar",
            lang,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        
        num_pairs = len(dataset['train'])
        total += num_pairs
        print(f"✓ Downloaded: {num_pairs:,} pairs")
        
    except Exception as e:
        print(f"✗ Failed: {lang} - {str(e)}")
        failed.append(lang)

print(f"\n{'='*70}")
print(f"Total: {total:,} pairs")
print(f"Success: {len(languages) - len(failed)}/{len(languages)}")
if failed:
    print(f"Failed: {', '.join(failed)}")
    sys.exit(1)

PYTHON_SCRIPT

    log "Samanantar complete!"
}

# INDICTRANS2 DOWNLOAD
download_indictrans2() {
    log "Starting IndicTrans2 models..."
    check_disk_space 20
    
    local models=(
        "ai4bharat/indictrans2-en-indic-1B"
        "ai4bharat/indictrans2-indic-en-1B"
        "ai4bharat/indictrans2-indic-indic-1B"
        "ai4bharat/indictrans2-en-indic-dist-200M"
        "ai4bharat/indictrans2-indic-en-dist-200M"
    )
    
    for model in "${models[@]}"; do
        log "Downloading: $model"
        download_with_retry "$model" || warning "Skipped: $model"
    done
    
    log "IndicTrans2 complete!"
}

# NLLB-200 DOWNLOAD
download_nllb() {
    log "Starting NLLB-200 models..."
    check_disk_space 50
    
    local models=(
        "facebook/nllb-200-distilled-600M"
        "facebook/nllb-200-1.3B"
        "facebook/nllb-200-3.3B"
    )
    
    for model in "${models[@]}"; do
        log "Downloading: $model"
        download_with_retry "$model" || warning "Skipped: $model"
    done
    
    log "NLLB-200 complete!"
}

# LLMS DOWNLOAD
download_llms() {
    log "Starting LLMs download..."
    check_disk_space 200
    
    local models=(
        "meta-llama/Llama-3.1-8B-Instruct"
        "mistralai/Mistral-7B-Instruct-v0.3"
        "mistralai/Mixtral-8x7B-Instruct-v0.1"
        "CohereForAI/aya-101"
        "bigscience/bloomz-7b1"
    )
    
    for model in "${models[@]}"; do
        log "Downloading: $model"
        download_with_retry "$model" || warning "Skipped: $model"
    done
    
    log "LLMs complete!"
}

# MAIN
main() {
    local option=${1:-all}
    
    log "=========================================="
    log "NLP Resources Download"
    log "=========================================="
    log "Option: $option"
    log "Base: $BASE_DIR"
    log "Log: $LOG_FILE"
    log "=========================================="
    
    case $option in
        samanantar)
            download_samanantar
            ;;
        indictrans|indictrans2)
            download_indictrans2
            ;;
        nllb)
            download_nllb
            ;;
        llms)
            download_llms
            ;;
        all)
            check_disk_space 600
            download_samanantar
            download_indictrans2
            download_nllb
            download_llms
            ;;
        *)
            error "Unknown option: $option"
            echo "Usage: $0 [all|samanantar|indictrans|nllb|llms]"
            exit 1
            ;;
    esac
    
    log "=========================================="
    log "Complete!"
    log "=========================================="
    
    df -h "$BASE_DIR" | tee -a "$LOG_FILE"
    du -sh "$BASE_DIR"/* | tee -a "$LOG_FILE"
}

main "$@"
```

Make executable:
```bash
chmod +x "$BASE_DIR/scripts/master_download.sh"
```

### Script 2: Samanantar Downloader

Save as `$BASE_DIR/scripts/download_samanantar.py`:

```python
#!/usr/bin/env python3
"""
Samanantar Corpus Downloader
Downloads English-Indic parallel corpus
"""

import os
import sys
from datasets import load_dataset
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'samanantar_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

CACHE_DIR = os.environ.get('HF_DATASETS_CACHE', './cache/datasets')
LANGUAGES = ['as', 'bn', 'gu', 'hi', 'kn', 'ml', 'mr', 'or', 'pa', 'ta', 'te']

def download_language(lang: str) -> tuple:
    """Download a single language pair"""
    logger.info(f"\n{'='*70}")
    logger.info(f"English ↔ {lang.upper()}")
    logger.info(f"{'='*70}")
    
    try:
        dataset = load_dataset(
            "ai4bharat/samanantar",
            lang,
            cache_dir=CACHE_DIR,
            trust_remote_code=True
        )
        
        num_pairs = len(dataset['train'])
        logger.info(f"✓ {num_pairs:,} pairs")
        
        # Sample
        sample = dataset['train'][0]
        logger.info(f"\nSample:")
        logger.info(f"  EN: {sample['src'][:100]}...")
        logger.info(f"  {lang.upper()}: {sample['tgt'][:100]}...")
        
        return True, num_pairs
        
    except Exception as e:
        logger.error(f"✗ Failed: {str(e)}")
        return False, 0

def main():
    logger.info("="*70)
    logger.info("SAMANANTAR DOWNLOAD")
    logger.info("="*70)
    
    total = 0
    successful = []
    failed = []
    
    for lang in LANGUAGES:
        success, num = download_language(lang)
        if success:
            successful.append(lang)
            total += num
        else:
            failed.append(lang)
    
    logger.info(f"\n{'='*70}")
    logger.info("SUMMARY")
    logger.info(f"{'='*70}")
    logger.info(f"Total: {total:,} pairs")
    logger.info(f"Success: {len(successful)}/{len(LANGUAGES)}")
    
    if failed:
        logger.warning(f"Failed: {', '.join(failed)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

Make executable:
```bash
chmod +x "$BASE_DIR/scripts/download_samanantar.py"
```

### Script 3: Verification Script

Save as `$BASE_DIR/scripts/verify_downloads.py`:

```python
#!/usr/bin/env python3
"""Verify downloaded resources"""

import os
import sys
from datasets import load_dataset
from transformers import AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_samanantar():
    """Verify Samanantar"""
    logger.info("\nChecking Samanantar...")
    
    cache_dir = os.environ.get('HF_DATASETS_CACHE')
    test_langs = ['hi', 'bn', 'ta']
    
    for lang in test_langs:
        try:
            dataset = load_dataset(
                "ai4bharat/samanantar",
                lang,
                cache_dir=cache_dir,
                trust_remote_code=True
            )
            logger.info(f"✓ {lang}: {len(dataset['train']):,} pairs")
        except Exception as e:
            logger.error(f"✗ {lang}: {e}")
            return False
    return True

def check_indictrans2():
    """Verify IndicTrans2"""
    logger.info("\nChecking IndicTrans2...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "ai4bharat/indictrans2-en-indic-dist-200M",
            trust_remote_code=True
        )
        logger.info("✓ IndicTrans2 accessible")
        return True
    except Exception as e:
        logger.error(f"✗ Failed: {e}")
        return False

def check_disk():
    """Check disk usage"""
    logger.info("\nDisk Usage:")
    
    base_dir = os.environ.get('BASE_DIR', '/mnt/data/nlp_resources')
    
    for root, dirs, files in os.walk(base_dir):
        size = sum(os.path.getsize(os.path.join(root, f)) 
                   for f in files if os.path.exists(os.path.join(root, f)))
        if size > 1e9:
            logger.info(f"  {root}: {size/1e9:.2f} GB")

def main():
    logger.info("="*70)
    logger.info("VERIFICATION")
    logger.info("="*70)
    
    checks = [
        ("Samanantar", check_samanantar),
        ("IndicTrans2", check_indictrans2),
    ]
    
    results = [check() for _, check in checks]
    check_disk()
    
    logger.info("\n" + "="*70)
    if all(results):
        logger.info("✓ ALL CHECKS PASSED")
    else:
        logger.error("✗ SOME FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

Make executable:
```bash
chmod +x "$BASE_DIR/scripts/verify_downloads.py"
```

---

## Implementation Recommendations

### Recommended Workflow

**Phase 1: Setup (30 minutes)**
```bash
# 1. Install Conda
cd /tmp
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc

# 2. Create environment
conda create -n NLPLResourceDownload python=3.10 -y
conda activate NLPLResourceDownload

# 3. Install packages
conda install -c conda-forge -y pytorch transformers datasets accelerate
pip install huggingface-hub hf-transfer

# 4. Setup directories
export BASE_DIR="/mnt/data/nlp_resources"
mkdir -p "$BASE_DIR"/{datasets,models,cache,logs,scripts}

# 5. Configure environment variables
export HF_DATASETS_CACHE="$BASE_DIR/cache/datasets"
export HF_HOME="$BASE_DIR/cache/huggingface"
export TRANSFORMERS_CACHE="$BASE_DIR/cache/transformers"

# 6. Login to HuggingFace
huggingface-cli login
```

**Phase 2: Download Resources (4-8 hours)**
```bash
cd "$BASE_DIR/scripts"

# Create master_download.sh (copy from above)
# Make executable
chmod +x master_download.sh

# Download everything
./master_download.sh all

# Or download selectively
./master_download.sh samanantar
./master_download.sh indictrans
```

**Phase 3: Verify (5 minutes)**
```bash
python verify_downloads.py
```

### For the 6 Unsupported Languages

**Option A: Pure Open Source (FREE)**
```python
# Use IndicTrans2 for Hindi pivot
# English → Hindi → Target Language
# Completely free, compute costs only
```

**Option B: Tiered with Claude API (₹62,912 / ~$740)**
```python
# 70% IndicTrans2 pivot (free)
# 20% Claude Haiku 3.5 enhancement
# 10% Claude Haiku 4.5 premium
# Use Batch API + Prompt Caching
```

### Disk Space Requirements

```
Minimum Setup (Essential):
  Samanantar (6 major langs): ~40 GB
  IndicTrans2 (distilled):    ~2 GB
  Working space:              ~8 GB
  Total:                      ~50 GB

Recommended Setup:
  Samanantar (all 11):        ~70 GB
  IndicTrans2 (all):          ~15 GB
  NLLB-200 (3.3B):           ~15 GB
  Working space:              ~20 GB
  Total:                      ~120 GB

Complete Setup:
  Samanantar:                 ~70 GB
  IndicTrans2:                ~15 GB
  NLLB-200 (54.5B):          ~210 GB
  LLMs (Llama/Mistral):      ~150 GB
  Working space:              ~50 GB
  Total:                      ~500 GB
```

---

## Quick Start Commands

```bash
# Activate environment
conda activate NLPLResourceDownload

# Download everything
cd /mnt/data/nlp_resources/scripts
./master_download.sh all

# Download selectively
./master_download.sh samanantar
./master_download.sh indictrans

# Verify
python verify_downloads.py

# Check disk usage
du -sh /mnt/data/nlp_resources/*
df -h /mnt/data/nlp_resources

# Monitor progress
tail -f /mnt/data/nlp_resources/logs/download_*.log
```

---

## References

### Papers
- **Samanantar**: https://arxiv.org/abs/2104.05596
- **IndicTrans2**: https://arxiv.org/abs/2305.16307
- **NLLB**: https://arxiv.org/abs/2207.04672
- **Aya-101**: https://arxiv.org/abs/2402.06619

### Resources
- **AI4Bharat**: https://ai4bharat.org/
- **Samanantar Dataset**: https://huggingface.co/datasets/ai4bharat/samanantar
- **IndicTrans2 Models**: https://huggingface.co/ai4bharat
- **Claude API Docs**: https://docs.anthropic.com/claude/docs
- **HuggingFace Datasets**: https://huggingface.co/docs/datasets/

### Key Environment Variables
```bash
BASE_DIR="/mnt/data/nlp_resources"
HF_DATASETS_CACHE="$BASE_DIR/cache/datasets"
HF_HOME="$BASE_DIR/cache/huggingface"
TRANSFORMERS_CACHE="$BASE_DIR/cache/transformers"
HF_HUB_ENABLE_HF_TRANSFER=1
```

---

## Summary

**Best Approach for 6 Unsupported Languages:**
- Use IndicTrans2 for free Hindi pivot translation (70-90% of corpus)
- Enhance 10-30% with Claude Haiku API for quality improvement
- Use Batch API + Prompt Caching for cost optimization
- **Total cost: ₹45,900 - ₹91,800 (~$540-$1,080)**
- **With full optimization: ₹62,912 (~$740)**

**Timeline:**
- Setup: 30 minutes
- Samanantar download: 2-4 hours
- Models download: 1-3 hours
- Total: 4-8 hours (depends on internet speed)

**Storage:**
- Minimum: 50 GB
- Recommended: 120 GB
- Complete: 500 GB

---

**Document Version:** 1.0  
**Last Updated:** November 12, 2025  
**Environment:** Ubuntu 24.04, Conda, Python 3.10