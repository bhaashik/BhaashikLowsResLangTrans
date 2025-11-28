# GPT-5 Nano Hindi Translation Package

Complete toolkit for translating Hindi to Bhojpuri and Magahi using GPT-5 Nano (ultra-fast, ultra-cheap).

## 📦 What's Included

This package contains:

1. **translate_domain_corpus.py** - Main translation script
2. **setup_wsl2.sh** - Automated setup for WSL2 Ubuntu
3. **requirements.txt** - Python dependencies
4. **README.md** - This file (complete instructions)

---

## ⚡ Quick Start (5 Minutes)

### For WSL2 Ubuntu with Conda:

```bash
# 1. Extract package
cd ~/
mkdir hindi_translation
cd hindi_translation

# 2. Copy all files here, then run setup
bash setup_wsl2.sh

# 3. The script will guide you through the setup!
```

### Manual Setup (Any Linux):

```bash
# 1. Activate conda environment
conda activate NLPLResourceDownload  # Or your env name

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set API key
export OPENAI_API_KEY="sk-proj-your-key-here"

# 4. Run translation
python translate_domain_corpus.py
```

---

## 📋 Requirements

### System Requirements:
- **OS**: Ubuntu 20.04+ (WSL2 or native), or any Linux
- **Python**: 3.8 or higher
- **Conda**: Miniconda or Anaconda (optional but recommended)
- **Disk Space**: 
  - Script: <1 MB
  - Output: Depends on your corpus size
  - Recommended: 10+ GB free space

### Python Dependencies:
- `openai` (Python SDK for OpenAI API)

That's it! Just one package.

---

## 🚀 Detailed Setup Instructions

### Step 1: Install Conda (if not already installed)

```bash
# Download Miniconda
cd /tmp
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Install
bash Miniconda3-latest-Linux-x86_64.sh

# Follow prompts, then reload shell
source ~/.bashrc

# Verify
conda --version
```

### Step 2: Create Conda Environment (optional)

```bash
# Create environment
conda create -n hindi_translation python=3.10 -y

# Activate
conda activate hindi_translation
```

### Step 3: Install Dependencies

```bash
# Using requirements.txt
pip install -r requirements.txt

# Or manually
pip install openai
```

### Step 4: Get OpenAI API Key

1. Go to: https://platform.openai.com/api-keys
2. Click "Create new secret key"
3. Copy the key (starts with `sk-proj-...`)
4. Set it:

```bash
# Temporary (current session only)
export OPENAI_API_KEY="sk-proj-your-key-here"

# Permanent (add to ~/.bashrc)
echo 'export OPENAI_API_KEY="sk-proj-your-key-here"' >> ~/.bashrc
source ~/.bashrc
```

### Step 5: Prepare Your Hindi Corpus

Create a domain-organized directory structure:

```
hindi_corpus/
├── news/
│   ├── article1.txt
│   ├── article2.txt
│   └── article3.txt
├── literature/
│   ├── story1.txt
│   └── poetry1.txt
└── social/
    ├── twitter.txt
    └── facebook.txt
```

Each `.txt` file should contain Hindi sentences (one per line):

```
यह पहला वाक्य है।
यह दूसरा वाक्य है।
यह तीसरा वाक्य है।
```

### Step 6: Run Translation

```bash
# Make script executable
chmod +x translate_domain_corpus.py

# Run
python translate_domain_corpus.py
```

The script will prompt you for:
- **Source directory**: Path to your `hindi_corpus/` directory
- **Output directory**: Where to save translations (default: `translations/`)
- **File pattern**: Which files to process (default: `*.txt`)
- **Target languages**: Which languages to translate to (default: `Bhojpuri,Magahi`)

---

## 📁 Output Structure

After translation, you'll get:

```
translations/
├── bhojpuri/
│   ├── news/
│   │   ├── article1.txt      ✅ Bhojpuri translation
│   │   ├── article2.txt
│   │   └── article3.txt
│   ├── literature/
│   │   ├── story1.txt
│   │   └── poetry1.txt
│   └── social/
│       ├── twitter.txt
│       └── facebook.txt
└── magahi/
    └── (same structure)
```

**Perfect mirror** of your source directory!

---

## 💰 Cost Information

### GPT-5 Nano Pricing (as of Nov 2025):
- **Input**: $0.05 per 1M tokens
- **Output**: $0.40 per 1M tokens

### Example Costs:

| Sentences | Cost per Language | Total (2 languages) |
|-----------|-------------------|---------------------|
| 10,000 | ₹26 (~$0.30) | ₹52 (~$0.60) |
| 35,000 | ₹91 (~$1.07) | ₹182 (~$2.14) |
| 70,000 | ₹183 (~$2.15) | ₹366 (~$4.30) |
| 100,000 | ₹261 (~$3.07) | ₹522 (~$6.14) |

**Including Maithili (FREE via IndicTrans2):**
- Total for 35,000 sentences × 3 languages: **₹182 (~$2.14)**

### Speed:
- **GPT-5 Nano**: Ultra-fast (~30-50 sentences/second)
- **70,000 sentences**: ~20-40 minutes total

---

## 🔧 Advanced Usage

### Translate Only Specific Languages:

When prompted, enter:
```
Translate to (comma-separated): Bhojpuri
```
Or:
```
Translate to (comma-separated): Magahi
```

### Custom File Patterns:

Process different file types:
```bash
# When prompted:
Enter file pattern: *.conllu   # For UD treebanks
Enter file pattern: *.tsv      # For tab-separated
Enter file pattern: *.md       # For markdown
```

### Batch Size Adjustment:

Edit `translate_domain_corpus.py` line ~167:

```python
# Default: 50 sentences per API call
batch_size: int = 50

# For longer sentences, reduce:
batch_size: int = 30

# For shorter sentences, increase:
batch_size: int = 100
```

---

## 🛡️ File Exclusions

The script automatically excludes temporary and backup files:

**Excluded extensions:**
- `.swp` - Vim swap files
- `.tmp` - Temporary files
- `.bak` - Backup files
- `.old` - Old versions
- `.backup` - Backup files
- `~` - Emacs backup files
- `.pyc` - Python compiled
- `.log` - Log files
- `.cache` - Cache files

**Also excluded:**
- Hidden files (starting with `.`)
- Any file ending with `~`

To add more exclusions, edit the `EXCLUDED_EXTENSIONS` set in the script (line ~42).

---

## 📊 Features

✅ **Domain preservation** - Maintains your directory structure  
✅ **Automatic file filtering** - Excludes temp/backup files  
✅ **Batch processing** - Efficient API usage (50 sentences/call)  
✅ **Progress tracking** - Real-time updates per file and domain  
✅ **Error handling** - Graceful recovery from failures  
✅ **Cost estimation** - Shows cost before starting  
✅ **Statistics** - Speed, errors, duration summary  
✅ **Linguistic expertise** - Bhojpuri/Magahi grammar rules built-in  
✅ **Resumable** - Can re-run safely (skips existing files)  

---

## 🔍 Linguistic Quality

The script includes expert-level linguistic prompts for:

### Bhojpuri (भोजपुरी):
- Proper copula usage: बा/बाटे/बानी
- Authentic pronouns: हम, रउवा
- Correct verb forms: रहल, गइल
- Natural vocabulary: बढ़िया, कालहि, etc.

### Magahi (मगही):
- Proper copula usage: छै/छथिन/छी
- Authentic pronouns: हम, तोहरा
- Correct verb forms: रहल, गेल
- Natural vocabulary: बढ़िया, नीक, etc.

Both maintain natural, colloquial style (not formal/literary).

---

## 🐛 Troubleshooting

### Error: "OPENAI_API_KEY not set"

**Solution:**
```bash
export OPENAI_API_KEY="sk-proj-your-key-here"

# Or add to ~/.bashrc for persistence
echo 'export OPENAI_API_KEY="sk-proj-your-key-here"' >> ~/.bashrc
source ~/.bashrc
```

### Error: "Missing openai package"

**Solution:**
```bash
pip install openai
```

### Error: "No valid files found"

**Solution:**
- Check your directory path is correct
- Ensure files have `.txt` extension (or specify correct pattern)
- Check files aren't all excluded (hidden, temp, backup)

### Slow file access on WSL2 /mnt/c/

**Solution:**
```bash
# Copy corpus to WSL2 filesystem (much faster)
cp -r /mnt/c/Users/YourName/hindi_corpus ~/
cd ~/
python translate_domain_corpus.py
```

### API Rate Limit Errors

**Solution:**
- The script includes automatic 0.1s delays between batches
- If still hitting limits, reduce batch size (edit line ~167)
- Wait a few minutes and re-run (script will resume)

### Permission Denied

**Solution:**
```bash
chmod +x translate_domain_corpus.py
chmod +x setup_wsl2.sh
```

---

## 📝 Example Session

```bash
$ python translate_domain_corpus.py

======================================================================
GPT-5 NANO DOMAIN CORPUS TRANSLATOR
Hindi → Bhojpuri & Magahi
Domain-organized with automatic temp file exclusion
======================================================================
✅ API key found

======================================================================
CONFIGURATION
======================================================================

Enter path to Hindi corpus directory: hindi_corpus
Enter output base directory [translations]: 
Enter file pattern [*.txt]: 

Translate to (comma-separated) [Bhojpuri,Magahi]: 

✅ Configuration complete
   Source: /home/user/hindi_corpus
   Output: /home/user/translations
   Pattern: *.txt
   Languages: Bhojpuri, Magahi
   Excluding: .bak, .cache, .log, .old, .pyc, .swp, .tmp, ~, backup

💰 COST ESTIMATE
======================================================================
Scanning corpus...
Valid files: 10
Excluded: 2
Counting sentences...
✅ Total sentences: 35,000

Per language (35,000 sentences):
  Cost: ₹91 (~$1.07)

Total for 2 language(s):
  Cost: ₹182 (~$2.14)
  Model: GPT-5 Nano (Ultra-Fast)
======================================================================

Proceed with translation? (y/n): y

[Translation proceeds with progress bars and statistics...]
```

---

## 🎯 Use Cases

### 1. Research Corpus Creation
Translate parallel corpora for low-resource NLP research in Bhojpuri and Magahi.

### 2. Dataset Augmentation
Create synthetic parallel data for training translation models.

### 3. Treebank Translation
Translate Universal Dependencies or other annotated corpora (preserves structure).

### 4. Document Translation
Translate domain-organized document collections while maintaining categorization.

### 5. Evaluation Set Creation
Build evaluation datasets for machine translation systems.

---

## 🤝 Support & Feedback

### Common Questions:

**Q: Can I translate to other languages?**  
A: Currently supports Bhojpuri and Magahi. To add more languages, edit the `SYSTEM_PROMPTS` dictionary in the script with appropriate linguistic rules.

**Q: Can I use a different model?**  
A: Yes! Change line ~86 from `"gpt-5-nano"` to:
- `"gpt-5-mini"` (better quality, higher cost)
- `"gpt-4o-mini"` (good alternative)
- `"gpt-4o"` (highest quality, expensive)

**Q: How do I add Maithili translation?**  
A: Maithili is FREE using IndicTrans2! See your project documents for setup instructions.

**Q: What about other Indic languages?**  
A: For scheduled languages (Hindi, Bengali, Tamil, etc.), use free IndicTrans2. For others, add prompts to this script.

---

## 📚 Additional Resources

- **OpenAI API Docs**: https://platform.openai.com/docs
- **GPT-5 Nano Info**: https://openai.com/gpt-5
- **Bhojpuri Wikipedia**: https://bh.wikipedia.org/
- **Magahi Resources**: https://en.wikipedia.org/wiki/Magahi_language

---

## 📄 License

These scripts are provided as-is for research and educational purposes.

**API Usage**: Subject to OpenAI's terms of service and pricing.

---

## ✅ Final Checklist

Before running:
- [ ] Conda environment created/activated
- [ ] `pip install openai` completed
- [ ] API key obtained from OpenAI
- [ ] API key set as environment variable
- [ ] Hindi corpus prepared in domain structure
- [ ] Scripts downloaded and made executable
- [ ] Sufficient disk space available

Ready to translate!

---

## 🚀 Quick Commands Reference

```bash
# Setup
bash setup_wsl2.sh

# Or manual
conda activate your_env
pip install openai
export OPENAI_API_KEY="sk-proj-..."

# Run
python translate_domain_corpus.py

# Test with example
mkdir -p hindi_corpus/test
echo "यह परीक्षण है।" > hindi_corpus/test/sample.txt
python translate_domain_corpus.py
```

---

**Version**: 1.0  
**Date**: November 2025  
**Model**: GPT-5 Nano  
**Languages**: Hindi → Bhojpuri, Magahi  

**Total Package Size**: ~50 KB (scripts only)  
**Documentation**: Complete  
**Ready to Use**: Yes ✅  

---

**Happy Translating! 🎉**

For questions or issues, refer to the troubleshooting section above.
