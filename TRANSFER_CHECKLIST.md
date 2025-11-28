# Transfer Checklist - Moving BhaashikLowsResLangTrans to Another Computer

Use this checklist to ensure complete transfer of the project to a new computer.

## Pre-Transfer Preparation (Source Computer)

### ☐ Verify Current State

```bash
# Check git status
cd BhaashikLowsResLangTrans
git status

# Commit any uncommitted changes
git add .
git commit -m "Pre-transfer commit"
git push origin main

# Check current conda environment
conda info --envs | grep bhaashik

# List installed packages
conda list > installed_packages_$(date +%Y%m%d).txt
pip list > pip_packages_$(date +%Y%m%d).txt
```

### ☐ Backup Critical Files

```bash
# Create backup directory
mkdir -p transfer_backup

# Copy configuration files
cp .env transfer_backup/env_backup.txt
cp -r universal_translate/config transfer_backup/
cp -r universal_translate/data transfer_backup/

# Backup any custom scripts
cp -r scripts/*.py transfer_backup/ 2>/dev/null || true

# Create archive (optional)
tar -czf bhaashik_backup_$(date +%Y%m%d).tar.gz transfer_backup/
```

### ☐ Document Custom Modifications

```bash
# List modified files
git status
git diff > custom_modifications_$(date +%Y%m%d).diff

# Document custom paths/settings
echo "Current working directory: $(pwd)" > transfer_backup/paths.txt
echo "Input directory: $(readlink -f input)" >> transfer_backup/paths.txt
echo "Output directory: $(readlink -f output)" >> transfer_backup/paths.txt
```

### ☐ Export Environment

```bash
# Export full conda environment
conda env export > environment_full_$(date +%Y%m%d).yml

# Export minimal environment (already in repo)
conda env export --from-history > environment.yml
```

### ☐ Package Data for Transfer

**Option A: Using Git (Recommended for code only)**
```bash
# Push all changes
git push origin main

# Note: Large data files should be transferred separately
```

**Option B: Using Archive (For complete transfer including data)**
```bash
# Create complete archive (WARNING: Can be 50+ GB!)
cd ..
tar -czf BhaashikLowsResLangTrans_complete.tar.gz \
  --exclude='BhaashikLowsResLangTrans/.git' \
  --exclude='BhaashikLowsResLangTrans/__pycache__' \
  --exclude='BhaashikLowsResLangTrans/*.pyc' \
  BhaashikLowsResLangTrans/

# Or exclude large output directory
tar -czf BhaashikLowsResLangTrans_no_output.tar.gz \
  --exclude='BhaashikLowsResLangTrans/output' \
  --exclude='BhaashikLowsResLangTrans/.git' \
  --exclude='BhaashikLowsResLangTrans/__pycache__' \
  BhaashikLowsResLangTrans/
```

**Option C: Using Cloud Storage**
```bash
# Upload to cloud (examples)
# Google Drive: Use rclone or gdrive
# Dropbox: Use Dropbox client
# AWS S3: aws s3 sync

# Example with AWS S3
aws s3 sync BhaashikLowsResLangTrans/ s3://your-bucket/bhaashik/ \
  --exclude ".git/*" --exclude "output/*"
```

---

## Transfer Methods

### Method 1: Git Clone (Best for code)

**On Source Computer:**
```bash
git push origin main
```

**On Target Computer:**
```bash
git clone https://github.com/bhaashik/BhaashikLowsResLangTrans.git
```

Then transfer data separately (see below).

### Method 2: USB/External Drive

```bash
# On source computer
cp BhaashikLowsResLangTrans_complete.tar.gz /media/usb/

# On target computer
cp /media/usb/BhaashikLowsResLangTrans_complete.tar.gz ~/
tar -xzf BhaashikLowsResLangTrans_complete.tar.gz
```

### Method 3: Network Transfer (SCP/RSYNC)

```bash
# From source to target computer (via SSH)
rsync -avz --progress \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  BhaashikLowsResLangTrans/ \
  user@target-computer:~/BhaashikLowsResLangTrans/

# Or use SCP
scp -r BhaashikLowsResLangTrans/ user@target-computer:~/
```

---

## Setup on Target Computer

### ☐ Step 1: Install Prerequisites

```bash
# Verify Python version
python3 --version  # Should be 3.10+

# Install Conda if not present
# See SETUP.md for installation instructions

# Verify Git
git --version
```

### ☐ Step 2: Extract/Clone Project

**If using Git:**
```bash
git clone https://github.com/bhaashik/BhaashikLowsResLangTrans.git
cd BhaashikLowsResLangTrans
```

**If using archive:**
```bash
tar -xzf BhaashikLowsResLangTrans_complete.tar.gz
cd BhaashikLowsResLangTrans
```

### ☐ Step 3: Create Conda Environment

```bash
# Create environment
conda env create -f environment.yml

# Activate
conda activate bhaashik-translation

# Verify
python --version
which python
```

### ☐ Step 4: Install Dependencies

```bash
# Install package in editable mode
pip install -e .

# Verify installation
python -c "from universal_translate import TranslationRequest; print('✓ Installed')"
```

### ☐ Step 5: Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit with your API keys
nano .env
# Or: vim .env
# Or: code .env  (VSCode)

# Add your OpenAI API key
# OPENAI_API_KEY=sk-your-key-here
```

### ☐ Step 6: Transfer Data Files

**If data wasn't included in transfer:**

```bash
# Create input directory structure
mkdir -p input/converted/Hindi/plain-text/{by_file,by_domain}
mkdir -p input/converted/Hindi/mappings

# Transfer input data from source computer
# Example using SCP:
scp -r user@source-computer:~/BhaashikLowsResLangTrans/input/ ./

# Or use cloud storage
# aws s3 sync s3://your-bucket/bhaashik/input/ input/
```

**Verify data:**
```bash
# Check Hindi input files
find input/converted/Hindi/plain-text/by_file -name "*.txt" | wc -l
# Should show 1432 files

# Check mappings
ls -la input/converted/Hindi/mappings/
```

### ☐ Step 7: Create Output Directories

```bash
# Create output structure
mkdir -p output/{Bhojpuri,Magahi,Maithili}/plain-text/by_file
mkdir -p logs
```

### ☐ Step 8: Verify Installation

```bash
# Run verification tests
python -c "
import sys
import openai
import conllu
import yaml
from universal_translate import TranslationRequest
print('✓ All imports successful')
print(f'✓ Python: {sys.version}')
print(f'✓ OpenAI: {openai.__version__}')
print(f'✓ CoNLL-U: {conllu.__version__}')
"

# Check API key
python -c "
from dotenv import load_dotenv
import os
load_dotenv()
key = os.getenv('OPENAI_API_KEY')
if key and key.startswith('sk-'):
    print('✓ API key configured')
else:
    print('✗ API key not found or invalid')
"
```

### ☐ Step 9: Run Test Translation

```bash
# Test with 1 file (~$0.002)
python scripts/translate_hindi_to_lowres_openai.py \
  --target-lang bho \
  --max-files 1

# Verify output
head -5 output/Bhojpuri/plain-text/by_file/AGRICULTURE/*.txt
```

---

## Post-Transfer Verification

### ☐ Functional Tests

```bash
# Test 1: Environment
conda info --envs | grep bhaashik

# Test 2: Package imports
python << 'PYEOF'
from universal_translate import TranslationRequest, TranslationUnit
from universal_translate.providers.openai_provider import OpenAIProvider
from universal_translate.prompts import PromptManager
print("✓ All imports working")
PYEOF

# Test 3: File access
ls -lh input/converted/Hindi/plain-text/by_file/ | head
ls -lh universal_translate/config/prompts/

# Test 4: Translation (1 file)
python scripts/translate_hindi_to_lowres_openai.py \
  --target-lang bho \
  --max-files 1
```

### ☐ Compare with Source Computer

```bash
# On target computer, generate checksums
find universal_translate/config -type f -name "*.yaml" -exec md5sum {} \; | sort > config_checksums.txt
find universal_translate/data -type f -name "*.json" -exec md5sum {} \; | sort > data_checksums.txt

# Compare with source computer's checksums
# (Transfer these files from source and use diff)
```

### ☐ Performance Test

```bash
# Test with 5 files (~$0.02, ~2-3 minutes)
time python scripts/translate_hindi_to_lowres_openai.py \
  --target-lang bho \
  --max-files 5

# Verify timing is reasonable
# Should complete in 2-5 minutes depending on internet speed
```

---

## Common Transfer Issues

### Issue: "conda: command not found"

**Solution:**
```bash
# Ensure conda is in PATH
export PATH="$HOME/miniconda3/bin:$PATH"
source ~/.bashrc

# Or reinstall Miniconda
```

### Issue: "No module named 'universal_translate'"

**Solution:**
```bash
# Reinstall package
conda activate bhaashik-translation
pip install -e .
```

### Issue: "Input files not found"

**Solution:**
```bash
# Verify input data was transferred
ls -la input/converted/Hindi/plain-text/by_file/

# If missing, transfer from source computer
scp -r user@source:~/BhaashikLowsResLangTrans/input/ ./
```

### Issue: "Permission denied" errors

**Solution:**
```bash
# Fix permissions
chmod -R u+w output/
chmod +x scripts/*.py
```

### Issue: Different Python version

**Solution:**
```bash
# Remove environment
conda env remove -n bhaashik-translation

# Edit environment.yml to match available Python
# Change: python=3.12 to python=3.11 or python=3.10

# Recreate environment
conda env create -f environment.yml
```

---

## Files to Transfer

### Essential Files (Must transfer)
- [ ] `environment.yml` ✓ (in repo)
- [ ] `pyproject.toml` ✓ (in repo)
- [ ] `requirements.txt` ✓ (in repo)
- [ ] `.env.example` ✓ (in repo)
- [ ] `universal_translate/` directory ✓ (in repo)
- [ ] `scripts/` directory ✓ (in repo)
- [ ] `input/` directory (large - transfer separately)

### Configuration Files (Must transfer)
- [ ] `.env` (contains API keys - transfer securely!)
- [ ] `universal_translate/config/prompts/*.yaml`
- [ ] `universal_translate/data/examples/*.json`

### Documentation (Should transfer)
- [ ] `README.md` ✓ (in repo)
- [ ] `SETUP.md` ✓ (in repo)
- [ ] `CLAUDE.md` ✓ (in repo)
- [ ] `FULL_TRANSLATION_GUIDE.md` ✓ (in repo)
- [ ] `GPT-4O-MINI-FIXED.md` ✓ (in repo)

### Optional Files
- [ ] `output/` (only if transferring existing translations)
- [ ] `logs/` (only if needed for reference)
- [ ] Custom modifications/patches

---

## Security Checklist

### ☐ Protect API Keys

```bash
# Ensure .env is not committed to git
grep '.env' .gitignore  # Should be listed

# Set proper permissions
chmod 600 .env

# Never share .env file publicly
```

### ☐ Clean Sensitive Data

```bash
# Before sharing/transferring, remove sensitive info
rm -f transfer_backup/env_backup.txt
rm -f .env  # Will use .env.example on target

# Check for hardcoded keys in code
grep -r "sk-" scripts/ universal_translate/
# Should return no results
```

---

## Quick Transfer Script

Save this as `quick_transfer.sh` on source computer:

```bash
#!/bin/bash
# Quick transfer preparation script

echo "=== BhaashikLowsResLangTrans Transfer Preparation ==="

# Commit changes
git add .
git commit -m "Pre-transfer commit $(date +%Y%m%d)"
git push origin main

# Create transfer package (without large data)
echo "Creating transfer package..."
tar -czf transfer_package_$(date +%Y%m%d).tar.gz \
  --exclude='output' \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.env' \
  .

echo "✓ Transfer package created: transfer_package_$(date +%Y%m%d).tar.gz"
echo ""
echo "Next steps:"
echo "1. Transfer this package to target computer"
echo "2. Follow TRANSFER_CHECKLIST.md on target computer"
echo "3. Don't forget to transfer .env file separately (securely!)"
echo "4. Transfer input/ directory (large - may take time)"
```

---

## Success Criteria

Transfer is complete when:

- ☐ Environment activates: `conda activate bhaashik-translation`
- ☐ All imports work: `python -c "from universal_translate import TranslationRequest"`
- ☐ API key configured: Check with `python -c "import os; print(os.getenv('OPENAI_API_KEY'))"`
- ☐ Input data present: `ls input/converted/Hindi/plain-text/by_file/ | wc -l` shows 1432
- ☐ Test translation succeeds: 1 file translates correctly
- ☐ Output files non-empty: Translations contain actual Bhojpuri text

**Final Test:**
```bash
python scripts/translate_hindi_to_lowres_openai.py --target-lang bho --max-files 1 && \
head -3 output/Bhojpuri/plain-text/by_file/AGRICULTURE/*.txt && \
echo "✓✓✓ TRANSFER SUCCESSFUL ✓✓✓"
```
