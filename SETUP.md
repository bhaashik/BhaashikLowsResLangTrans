# BhaashikLowsResLangTrans - Setup Guide

Complete guide for setting up this project on a new computer.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Setup (5 minutes)](#quick-setup)
3. [Detailed Setup](#detailed-setup)
4. [Verification](#verification)
5. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software

- **Python 3.10+** (3.12 recommended)
- **Conda** (Miniconda or Anaconda)
- **Git** (for cloning the repository)
- **50+ GB free disk space** (for input/output data)

### Optional

- **Screen/Tmux** (for long-running translations)

### Get Prerequisites

**Windows:**
1. Install Miniconda: https://docs.conda.io/en/latest/miniconda.html
2. Install Git: https://git-scm.com/download/win

**Linux/WSL:**
```bash
# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Git (usually pre-installed)
sudo apt-get install git
```

**macOS:**
```bash
# Install Homebrew if not installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Miniconda
brew install miniconda

# Git (usually pre-installed)
brew install git
```

---

## Quick Setup (5 minutes)

For users who just want to get started quickly:

```bash
# 1. Clone the repository
git clone https://github.com/bhaashik/BhaashikLowsResLangTrans.git
cd BhaashikLowsResLangTrans

# 2. Create Conda environment
conda env create -f environment.yml

# 3. Activate environment
conda activate bhaashik-translation

# 4. Configure API key
cp .env.example .env
nano .env  # Add your OPENAI_API_KEY

# 5. Test installation
python scripts/translate_hindi_to_lowres_openai.py \
  --target-lang bho \
  --max-files 1

# If successful, you'll see translated output!
```

---

## Detailed Setup

### Step 1: Clone Repository

```bash
# Option A: HTTPS (recommended for most users)
git clone https://github.com/bhaashik/BhaashikLowsResLangTrans.git
cd BhaashikLowsResLangTrans

# Option B: SSH (if you have SSH keys configured)
git clone git@github.com:bhaashik/BhaashikLowsResLangTrans.git
cd BhaashikLowsResLangTrans

# Check you're on the main branch
git branch
```

### Step 2: Create Python Environment

#### Option A: Using Conda (Recommended)

```bash
# Create environment from YAML file
conda env create -f environment.yml

# Activate the environment
conda activate bhaashik-translation

# Verify installation
python --version  # Should show Python 3.12.x
which python     # Should point to conda environment
```

#### Option B: Using venv (Alternative)

```bash
# Create virtual environment
python3.12 -m venv venv

# Activate environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Step 3: Configure Environment Variables

```bash
# Copy template
cp .env.example .env

# Edit configuration
nano .env  # or use your preferred editor
```

**Required configuration:**
```bash
# Add your OpenAI API key
OPENAI_API_KEY=sk-your-actual-key-here
```

Get your API key from: https://platform.openai.com/api-keys

**Optional configurations:**
```bash
# Customize paths if needed
# PROJECT_BASE_DIR=/custom/path/to/project
# INPUT_DIR=custom_input
# OUTPUT_DIR=custom_output

# Adjust translation settings
# OPENAI_MODEL=gpt-4o-mini
# BATCH_SIZE=20
```

### Step 4: Install Universal Translate Package

```bash
# Install in editable mode (for development)
pip install -e .

# Or install normally
# pip install .

# Verify installation
python -c "from universal_translate import TranslationRequest; print('✓ Package installed')"
```

### Step 5: Verify Data Structure

Ensure input data is in place:

```bash
# Check directory structure
tree -L 3 -d

# Expected structure:
# .
# ├── input/
# │   └── converted/
# │       └── Hindi/
# │           ├── plain-text/
# │           │   ├── by_file/
# │           │   └── by_domain/
# │           └── mappings/
# ├── output/
# ├── universal_translate/
# ├── scripts/
# └── ...

# Verify Hindi input files exist
find input/converted/Hindi/plain-text/by_file -name "*.txt" | head -5
```

If input data is missing, you need to:
1. Get the original CoNLL-U files
2. Run the extraction script: `python scripts/extract_plaintext_from_conllu.py`

---

## Verification

### Test 1: Python Environment

```bash
conda activate bhaashik-translation

python << 'PYEOF'
import sys
print(f"✓ Python {sys.version}")

# Test imports
try:
    import openai
    print(f"✓ OpenAI version: {openai.__version__}")
except ImportError as e:
    print(f"✗ OpenAI import failed: {e}")

try:
    import conllu
    print(f"✓ CoNLL-U version: {conllu.__version__}")
except ImportError as e:
    print(f"✗ CoNLL-U import failed: {e}")

try:
    import yaml
    print(f"✓ PyYAML installed")
except ImportError as e:
    print(f"✗ PyYAML import failed: {e}")

try:
    from universal_translate import TranslationRequest
    print(f"✓ universal_translate package installed")
except ImportError as e:
    print(f"✗ universal_translate import failed: {e}")

print("\nAll checks passed! ✓")
PYEOF
```

### Test 2: API Key Configuration

```bash
python << 'PYEOF'
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')
if api_key and api_key.startswith('sk-'):
    print("✓ OpenAI API key configured")
    print(f"  Key starts with: {api_key[:15]}...")
else:
    print("✗ OpenAI API key not found or invalid")
    print("  Please set OPENAI_API_KEY in .env file")
PYEOF
```

### Test 3: Translation Pipeline

```bash
# Test with 1 file (~$0.002 cost)
python scripts/translate_hindi_to_lowres_openai.py \
  --target-lang bho \
  --max-files 1

# Check output was created
ls -lh output/Bhojpuri/plain-text/by_file/AGRICULTURE/ | head -5

# Verify translations are not empty
head -3 output/Bhojpuri/plain-text/by_file/AGRICULTURE/*.txt
```

Expected output:
```
सन 1967 में हमार वैज्ञानिकन ने कपास के पैदावार में बढ़ोत्‍तरी करे खातिर...
भारत में कपास के खेती के क्षेत्र दुनिया में सबसे अधिक...
किंतु उत्पादन में भारत के स्थान अमेरिका, रूस आ चीन...
```

---

## Troubleshooting

### Problem: Conda environment creation fails

**Symptoms:**
```
CondaHTTPError: HTTP 000 CONNECTION FAILED
```

**Solution:**
```bash
# Update conda
conda update -n base -c defaults conda

# Try creating environment again
conda env create -f environment.yml --force
```

### Problem: "ModuleNotFoundError: No module named 'openai'"

**Solution:**
```bash
# Make sure environment is activated
conda activate bhaashik-translation

# Reinstall dependencies
pip install -r requirements.txt

# Verify
python -c "import openai; print(openai.__version__)"
```

### Problem: "OpenAI API key not provided"

**Solution:**
```bash
# Check .env file exists
ls -la .env

# Check API key is set
cat .env | grep OPENAI_API_KEY

# If not set, edit .env
nano .env
# Add: OPENAI_API_KEY=sk-your-key-here

# Verify it loads
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print(os.getenv('OPENAI_API_KEY'))"
```

### Problem: Empty translation output files

**Symptoms:**
- Files are created but contain only blank lines
- Cost shows $0.00

**Solution:**
This is fixed in the current version. Make sure you're using **GPT-4o-mini** (default model):

```bash
# Should work correctly
python scripts/translate_hindi_to_lowres_openai.py --target-lang bho --max-files 1

# If you tried GPT-5-nano, use GPT-4o-mini instead
python scripts/translate_hindi_to_lowres_openai.py --target-lang bho --model gpt-4o-mini
```

See `GPT-4O-MINI-FIXED.md` for details about the fix.

### Problem: Rate limit errors

**Symptoms:**
```
Error: Rate limit exceeded
```

**Solution:**
```bash
# Option 1: Reduce batch size
python scripts/translate_hindi_to_lowres_openai.py \
  --target-lang bho \
  --batch-size 5

# Option 2: Upgrade your OpenAI account tier
# Visit: https://platform.openai.com/account/billing

# Option 3: Add delays between batches (requires code modification)
```

### Problem: Permission denied on Linux/Mac

**Symptoms:**
```
PermissionError: [Errno 13] Permission denied: 'output/...'
```

**Solution:**
```bash
# Fix directory permissions
chmod -R u+w output/

# Or recreate output directory
rm -rf output
mkdir -p output
```

### Problem: Input files not found

**Symptoms:**
```
✗ No .txt files found in input/converted/Hindi/plain-text/by_file
```

**Solution:**
```bash
# Check if input directory exists
ls -la input/converted/Hindi/plain-text/

# If missing, you need to extract from CoNLL-U files
python scripts/extract_plaintext_from_conllu.py

# Or restore from backup/original source
```

---

## Next Steps

Once setup is complete:

1. **Read usage documentation:**
   - `QUICK_START_OPENAI.md` - Quick reference
   - `FULL_TRANSLATION_GUIDE.md` - Complete guide
   - `OPENAI_TRANSLATION_USAGE.md` - Detailed usage

2. **Test with small dataset:**
   ```bash
   python scripts/translate_hindi_to_lowres_openai.py --target-lang bho --max-files 5
   ```

3. **Review sample translations:**
   ```bash
   head -10 output/Bhojpuri/plain-text/by_file/AGRICULTURE/*.txt
   ```

4. **Start full translation:**
   ```bash
   # For one language (~$4, ~8-10 hours)
   python scripts/translate_hindi_to_lowres_openai.py --target-lang bho
   
   # For all 3 languages (~$12, ~24-30 hours sequential)
   for lang in bho mag mai; do
       python scripts/translate_hindi_to_lowres_openai.py --target-lang $lang
   done
   ```

---

## Additional Resources

- **Project documentation:** `README.md`
- **Architecture guide:** `CLAUDE.md`
- **Translation package design:** `TRANSLATION_PACKAGE_DESIGN.md`
- **Migration guide:** `universal_translate/MIGRATION_GUIDE.md`
- **GPT-4o-mini fix details:** `GPT-4O-MINI-FIXED.md`

---

## Support

If you encounter issues not covered in this guide:

1. Check existing documentation in the project
2. Review error messages carefully
3. Verify your environment matches prerequisites
4. Check OpenAI API status: https://status.openai.com/

## License

This project is licensed under the MIT License. See `LICENSE` file for details.
