# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BhaashikLowsResLangTrans is a project for creating synthetic parallel corpora for low-resource Indic languages, particularly those not supported by IndicTrans2's 22 scheduled languages. The project focuses on cost-effective translation strategies using a combination of open-source models and commercial APIs.

### Target Languages

**Supported by IndicTrans2 (FREE - 22 languages):**
Assamese, Bengali, Gujarati, Hindi, Kannada, Kashmiri, Konkani, Malayalam, Manipuri, Marathi, Nepali, Odia, Punjabi, Sanskrit, Sindhi, Tamil, Telugu, Urdu, Bodo, Santhali, Maithili, Dogri

**NOT Supported (6 languages requiring budget):**
1. Bhojpuri (Indo-Aryan, ~51M speakers)
2. Magahi (Indo-Aryan, ~13M speakers)
3. Awadhi (Indo-Aryan, ~38M speakers)
4. Braj (Indo-Aryan, ~1M speakers)
5. Marwari (Indo-Aryan, ~13M speakers)
6. Bundeli (Indo-Aryan, ~3M speakers)

## Development Environment

### System Requirements
- **Platform**: Ubuntu 24.04 or later
- **Python**: 3.10+
- **Disk Space**: 120 GB minimum (150+ GB recommended for full setup)
- **RAM**: 16 GB minimum (32 GB recommended)
- **Internet**: 50+ Mbps recommended for downloading large datasets

### Environment Setup

```bash
# Create and activate conda environment
conda create -n NLPLResourceDownload python=3.10 -y
conda activate NLPLResourceDownload

# Install core packages
conda install -c conda-forge -y \
    pytorch torchvision torchaudio pytorch-cuda=11.8 \
    transformers datasets accelerate sentencepiece protobuf \
    numpy pandas tqdm requests aiohttp pyyaml

# Install additional packages
pip install huggingface-hub hf-transfer python-dotenv colorama rich

# For CPU-only systems
conda install -c conda-forge pytorch torchvision torchaudio cpuonly -y
```

### Environment Variables

These environment variables must be set for the project:

```bash
export BASE_DIR="/mnt/data/nlp_resources"  # Adjust to your preferred location
export HF_DATASETS_CACHE="$BASE_DIR/cache/datasets"
export HF_HOME="$BASE_DIR/cache/huggingface"
export TRANSFORMERS_CACHE="$BASE_DIR/cache/transformers"
export HF_HUB_ENABLE_HF_TRANSFER=1
```

Add these to `~/.bashrc` to make them persistent.

### Directory Structure

```
$BASE_DIR/
├── datasets/       # Downloaded parallel corpora
├── models/         # Translation models
├── cache/          # HuggingFace cache
│   ├── datasets/
│   ├── huggingface/
│   └── transformers/
├── logs/           # Download and processing logs
└── scripts/        # Download and processing scripts
```

## Data Sources

### Samanantar Corpus
- **Total pairs**: 49.7 million sentence pairs
- **Languages**: 11 Indic languages paired with English
- **Languages available**: as (Assamese), bn (Bengali), gu (Gujarati), hi (Hindi), kn (Kannada), ml (Malayalam), mr (Marathi), or (Odia), pa (Punjabi), ta (Tamil), te (Telugu)
- **Size**: ~25-30 GB compressed, ~50-60 GB uncompressed
- **Source**: `ai4bharat/samanantar` on HuggingFace

### Translation Models

**IndicTrans2 (FREE - Primary)**
- `ai4bharat/indictrans2-en-indic-1B` - English to Indic
- `ai4bharat/indictrans2-indic-en-1B` - Indic to English
- `ai4bharat/indictrans2-indic-indic-1B` - Indic to Indic
- `ai4bharat/indictrans2-en-indic-dist-200M` - Distilled English to Indic
- `ai4bharat/indictrans2-indic-en-dist-200M` - Distilled Indic to English

**NLLB-200 (FREE - Secondary)**
- `facebook/nllb-200-distilled-600M` - Lightweight
- `facebook/nllb-200-1.3B` - Balanced
- `facebook/nllb-200-3.3B` - High quality

**LLMs (FREE - Experimental)**
- `meta-llama/Llama-3.1-8B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`
- `CohereForAI/aya-101`

## Translation Strategy

### For 22 Supported Languages
Use IndicTrans2 directly - completely free (compute costs only).

### For 6 Unsupported Languages (Bhojpuri, Magahi, Awadhi, Braj, Marwari, Bundeli)

**Recommended Approach: Tiered Strategy**

1. **70% IndicTrans2 Hindi Pivot (FREE)**
   - English → Hindi → Target Language
   - Hindi is closely related to these languages

2. **20% Claude Haiku 3.5 Enhancement**
   - For quality improvement on challenging segments
   - Cost: ₹68 input / ₹340 output per 1M tokens

3. **10% Claude Haiku 4.5 Premium**
   - For domain-specific or culturally sensitive content
   - Cost: ₹85 input / ₹425 output per 1M tokens

**Cost Optimization:**
- Use Batch API (50% discount on output)
- Enable Prompt Caching (90% discount on cached input)
- **Estimated cost**: ₹62,912 (~$740) for all 6 languages (12M sentence pairs)
- **Per-language cost**: ₹10,485 (~$123) for 2M sentence pairs

## Common Commands

### Downloading Data

```bash
# Activate environment
conda activate NLPLResourceDownload

# Navigate to scripts directory
cd "$BASE_DIR/scripts"

# Download all resources (requires 600GB space)
./master_download.sh all

# Download selectively
./master_download.sh samanantar
./master_download.sh indictrans
./master_download.sh nllb
./master_download.sh llms

# Verify downloads
python verify_downloads.py
```

### Monitoring Downloads

```bash
# Check disk usage
du -sh "$BASE_DIR"/*
df -h "$BASE_DIR"

# Monitor download logs
tail -f "$BASE_DIR/logs/download_*.log"
```

### HuggingFace Authentication

Required for gated models (e.g., Llama):
```bash
huggingface-cli login
# Get token from: https://huggingface.co/settings/tokens
```

## Architecture Principles

### Translation Pipeline Design

1. **Data Loading**: Use HuggingFace `datasets` library with proper caching
2. **Model Selection**: Choose based on language pair and quality requirements
3. **Batch Processing**: Process in batches to manage memory
4. **Quality Control**: Implement validation and post-editing workflows
5. **Cost Tracking**: Monitor API usage for commercial services

### Hindi Pivot Strategy

For unsupported languages, use Hindi as a pivot language because:
- Hindi is linguistically related to all 6 target languages (Indo-Aryan family)
- IndicTrans2 has excellent Hindi support
- Reduces translation cost while maintaining reasonable quality

### Quality Enhancement

- Use open-source models for bulk translation (70-90%)
- Apply commercial APIs selectively for:
  - Complex sentences
  - Domain-specific content
  - Cultural nuances
  - Idiomatic expressions

## Cost Considerations

### Pricing (Indian Rupees)

**Claude API (per 1M tokens):**
- Haiku 3: ₹21 input / ₹106 output
- Haiku 3.5: ₹68 input / ₹340 output
- Haiku 4.5: ₹85 input / ₹425 output
- Sonnet 4.5: ₹255 input / ₹1,275 output

**Batch API**: 50% discount on output tokens
**Prompt Caching**: 90% discount on cached input tokens

### Budget Estimation

For 1.2 billion characters (12M sentence pairs) across 6 languages:
- Pure IndicTrans2: ₹0 (compute only)
- 90% pivot + 10% Haiku 4.5: ₹45,900 ($540)
- 80% pivot + 20% Haiku 4.5: ₹91,800 ($1,080)
- Tiered (70/20/10) with Batch + Cache: ₹62,912 ($740) **RECOMMENDED**

## References

### Key Papers
- **Samanantar**: https://arxiv.org/abs/2104.05596
- **IndicTrans2**: https://arxiv.org/abs/2305.16307
- **NLLB**: https://arxiv.org/abs/2207.04672

### Documentation
- **AI4Bharat**: https://ai4bharat.org/
- **Samanantar Dataset**: https://huggingface.co/datasets/ai4bharat/samanantar
- **IndicTrans2 Models**: https://huggingface.co/ai4bharat
- **Claude API**: https://docs.anthropic.com/claude/docs
- **HuggingFace Datasets**: https://huggingface.co/docs/datasets/

## Project Status

**Current Phase**: Initial setup and planning
**Next Steps**:
1. Create directory structure
2. Set up download scripts
3. Download Samanantar corpus
4. Download IndicTrans2 models
5. Implement Hindi pivot translation pipeline
6. Develop quality assessment metrics
7. Implement selective API enhancement
