# BhaashikLowsResLangTrans - Setup Guide

Complete setup guide for the low-resource language translation system.

## Quick Start

```bash
# 1. Clone the repository
git clone <repository-url>
cd BhaashikLowsResLangTrans

# 2. Create conda environment
conda env create -f environment.yml
conda activate NLPLResourceDownload

# 3. Install additional packages
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your settings

# 5. Verify setup
python scripts/verify.py --all

# 6. Download essential resources
python scripts/download.py --essential

# 7. Test translation
python scripts/translate.py --src en --tgt hi --text "Hello, how are you?" --indictrans2
```

## Detailed Setup

### 1. System Requirements

- **OS**: Ubuntu 20.04+ (or WSL2 on Windows)
- **Python**: 3.10+
- **Disk Space**:
  - Minimum: 50 GB (essential resources only)
  - Recommended: 120 GB (full Samanantar + models)
  - Complete: 500 GB (includes LLMs)
- **RAM**: 16 GB minimum, 32 GB recommended
- **GPU**: Optional but recommended (CUDA-compatible)

### 2. Install Conda

If you don't have Conda installed:

```bash
# Download Miniconda
cd /tmp
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Install
bash Miniconda3-latest-Linux-x86_64.sh

# Reload shell
source ~/.bashrc

# Verify
conda --version
```

### 3. Create Environment

#### Option A: Using environment.yml (Recommended)

```bash
conda env create -f environment.yml
conda activate NLPLResourceDownload
```

#### Option B: Manual Setup

```bash
# Create environment
conda create -n NLPLResourceDownload python=3.10 -y
conda activate NLPLResourceDownload

# Install PyTorch (with CUDA)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Or for CPU only
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# Install other packages
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
# Copy example env file
cp .env.example .env

# Edit .env file
nano .env
```

Required settings in `.env`:

```bash
# Base directory - CHANGE THIS to your preferred location
BASE_DIR=/mnt/data/nlp_resources

# HuggingFace token (required for Llama and gated models)
HF_TOKEN=your_token_here

# Claude API key (optional, only if using API enhancement)
ANTHROPIC_API_KEY=your_key_here
```

Get your HuggingFace token from: https://huggingface.co/settings/tokens

### 5. Create Directory Structure

The system will create directories automatically, but you can create them manually:

```bash
# Set your base directory
export BASE_DIR="/mnt/data/nlp_resources"

# Create structure
mkdir -p "$BASE_DIR"/{datasets,models,cache,logs,output,checkpoints}
mkdir -p "$BASE_DIR"/cache/{datasets,huggingface,transformers}
```

### 6. Download Resources

#### Essential Resources (~70 GB)

Includes Samanantar major languages and IndicTrans2 models:

```bash
python scripts/download.py --essential
```

#### Specific Languages

Download only the languages you need:

```bash
# Download specific languages
python scripts/download.py --samanantar --languages hi bn ta te mr gu

# Download all Samanantar languages
python scripts/download.py --samanantar --all-languages
```

#### Models Only

```bash
# IndicTrans2 models
python scripts/download.py --models indictrans2

# NLLB models
python scripts/download.py --models nllb

# Both
python scripts/download.py --models indictrans2 nllb
```

#### LLMs (Optional, ~200 GB)

```bash
python scripts/download.py --models llama mistral aya
```

### 7. Verify Installation

```bash
# Run all checks
python scripts/verify.py --all

# Check specific components
python scripts/verify.py --environment
python scripts/verify.py --directories
python scripts/verify.py --samanantar --languages hi bn ta
python scripts/verify.py --models
python scripts/verify.py --api-keys
```

## Configuration

### Main Configuration File

Edit `config/config.yaml` to customize:

- Language settings
- Model paths
- Translation strategies
- Cost tracking
- Processing parameters

### Tier Percentages

Default: 70% free, 20% enhancement, 10% premium

To change, edit `config/config.yaml`:

```yaml
strategy:
  unsupported_strategy:
    tiers:
      - name: "free_pivot"
        percentage: 80  # Change this
      - name: "quality_enhancement"
        percentage: 15  # Change this
      - name: "premium_quality"
        percentage: 5   # Change this
```

## Testing

### Test IndicTrans2 (Free)

```bash
# English to Hindi
python scripts/translate.py \
  --src en \
  --tgt hi \
  --text "Hello, how are you?" \
  --indictrans2

# Hindi to Bengali
python scripts/translate.py \
  --src hi \
  --tgt bn \
  --text "आप कैसे हैं?" \
  --indictrans2
```

### Test Hindi Pivot (Free)

```bash
# English to Bhojpuri (via Hindi)
python scripts/translate.py \
  --src en \
  --tgt bho \
  --text "Hello, how are you?" \
  --hindi-pivot
```

### Test Tiered Strategy (Costs Money)

```bash
# First, get cost estimate
python scripts/translate.py \
  --src en \
  --tgt bho \
  --num-samples 1000 \
  --estimate-only

# Then translate
python scripts/translate.py \
  --src en \
  --tgt bho \
  --input input.txt \
  --output output.txt \
  --tiered
```

## Troubleshooting

### CUDA Out of Memory

Reduce batch size:

```bash
python scripts/translate.py ... --batch-size 8
```

Or use CPU:

```bash
# Edit config/config.yaml
processing:
  use_gpu: false
  device: "cpu"
```

### HuggingFace Login Issues

```bash
# Login manually
huggingface-cli login

# Or set token in .env
HF_TOKEN=your_token_here
```

### Download Failures

```bash
# Clear cache and retry
rm -rf $BASE_DIR/cache/*
python scripts/download.py --essential
```

### Import Errors

```bash
# Reinstall packages
pip install --upgrade -r requirements.txt
```

## Next Steps

1. Read [USAGE.md](USAGE.md) for detailed usage examples
2. Review [CLAUDE.md](CLAUDE.md) for architecture details
3. Explore example notebooks (coming soon)
4. Check cost tracking: `cat $BASE_DIR/logs/cost_tracking.json`

## Support

- GitHub Issues: [Create an issue](../../issues)
- Documentation: [CLAUDE.md](CLAUDE.md)
- Planning Document: [Initial instructions](Initial%20intstructions%20(a%20log%20of%20a%20long%20conversation)%20for%20Claude%20Code.md)
