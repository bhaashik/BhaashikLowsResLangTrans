# Troubleshooting Guide - Hindi Translation

## Common Issues & Solutions

### 🔴 Installation Issues

#### Error: "conda: command not found"

**Cause**: Conda not installed or not in PATH

**Solution**:
```bash
# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Reload shell
source ~/.bashrc

# Verify
conda --version
```

#### Error: "pip: command not found"

**Cause**: Python/pip not in PATH

**Solution**:
```bash
# Make sure conda env is activated
conda activate NLPLResourceDownload

# Or install pip
sudo apt-get update
sudo apt-get install python3-pip
```

#### Error: "No module named 'openai'"

**Cause**: OpenAI package not installed

**Solution**:
```bash
# Install via pip
pip install openai

# Verify
python -c "import openai; print('OK')"
```

---

### 🔴 API Key Issues

#### Error: "OPENAI_API_KEY not set"

**Cause**: Environment variable not configured

**Solution**:
```bash
# Temporary (current session)
export OPENAI_API_KEY="sk-proj-your-key-here"

# Permanent (add to ~/.bashrc)
echo 'export OPENAI_API_KEY="sk-proj-your-key-here"' >> ~/.bashrc
source ~/.bashrc

# Verify
echo $OPENAI_API_KEY
```

#### Error: "Incorrect API key provided"

**Cause**: Wrong API key or typo

**Solution**:
1. Go to: https://platform.openai.com/api-keys
2. Generate new key
3. Copy carefully (starts with `sk-proj-`)
4. Set again: `export OPENAI_API_KEY="sk-proj-..."`

#### Error: "You exceeded your current quota"

**Cause**: No credits on OpenAI account

**Solution**:
1. Go to: https://platform.openai.com/account/billing
2. Add payment method
3. Purchase at least $5 credits
4. Wait ~5 minutes for activation

---

### 🔴 File & Directory Issues

#### Error: "Directory not found"

**Cause**: Wrong path or directory doesn't exist

**Solution**:
```bash
# Check current directory
pwd

# List contents
ls -la

# Create directory if needed
mkdir -p hindi_corpus/news

# Use absolute path
python translate_domain_corpus.py
# Then enter full path: /home/user/hindi_corpus
```

#### Error: "No valid files found"

**Cause**: Files excluded or wrong extension

**Solutions**:
```bash
# Check what files exist
find hindi_corpus -type f

# Check file extensions
ls -la hindi_corpus/*

# If files are .swp, .bak, etc. - they're excluded by design
# Rename them:
for f in hindi_corpus/*.bak; do mv "$f" "${f%.bak}.txt"; done
```

#### Error: "Permission denied"

**Cause**: File/directory permissions

**Solution**:
```bash
# Make script executable
chmod +x translate_domain_corpus.py

# Fix directory permissions
chmod -R 755 hindi_corpus/

# If copied from Windows
dos2unix translate_domain_corpus.py  # If line endings wrong
```

---

### 🔴 WSL2-Specific Issues

#### Slow file access from /mnt/c/

**Cause**: Cross-filesystem access is slow in WSL2

**Solution**:
```bash
# Copy to WSL2 filesystem (much faster)
cp -r /mnt/c/Users/YourName/hindi_corpus ~/
cd ~/
python translate_domain_corpus.py
```

**Speed difference**:
- `/mnt/c/`: ~1-2 MB/s (slow)
- `~/`: ~100-200 MB/s (fast)

#### Network/DNS issues

**Cause**: WSL2 DNS configuration

**Solution**:
```bash
# Check DNS
cat /etc/resolv.conf

# Fix DNS (use Google DNS)
sudo bash -c 'echo "nameserver 8.8.8.8" > /etc/resolv.conf'

# Test
ping google.com

# Make persistent (in /etc/wsl.conf)
sudo nano /etc/wsl.conf

# Add:
[network]
generateResolvConf = false
```

#### Can't find conda environment

**Cause**: Conda not initialized for bash

**Solution**:
```bash
# Initialize conda
~/miniconda3/bin/conda init bash

# Reload shell
source ~/.bashrc

# Now conda activate should work
conda activate NLPLResourceDownload
```

---

### 🔴 Translation Issues

#### Error: "Rate limit exceeded"

**Cause**: Too many API requests too fast

**Solution**:
```bash
# Script has built-in 0.1s delays
# If still failing, edit script line ~233:
time.sleep(0.5)  # Increase from 0.1 to 0.5

# Or wait 1 minute and re-run (script resumes)
```

#### Error: "Context length exceeded"

**Cause**: Sentences too long for batch

**Solution**:
```bash
# Edit script line ~167 to reduce batch size:
batch_size: int = 30  # From 50 to 30

# Or line ~20:
max_tokens=3000  # From 2000 to 3000
```

#### Translation quality issues

**Cause**: Model limitations for specific content

**Solutions**:
1. **Use better model** (edit line ~86):
   ```python
   self.model = "gpt-5-mini"  # Better quality, higher cost
   ```

2. **Adjust temperature** (edit line ~217):
   ```python
   temperature=0.1  # From 0.3 to 0.1 (more consistent)
   ```

3. **Add domain-specific examples** to SYSTEM_PROMPTS

#### Output has "[MISSING TRANSLATION]"

**Cause**: API call failed or response parsing error

**Solution**:
```bash
# Re-run - script will retry failed files
python translate_domain_corpus.py

# Check logs for specific error
# Look for "Error in batch" messages

# If persistent, reduce batch size (see above)
```

---

### 🔴 Cost Issues

#### Higher cost than expected

**Causes & Solutions**:

1. **Longer sentences**:
   - Hindi: ~60 chars/sentence (average)
   - If yours are 120+ chars: Cost doubles
   - Solution: Nothing wrong, just reality

2. **Failed translations retried**:
   - Script retries failures
   - Solution: Check error count in summary

3. **Used wrong model**:
   - Check line ~86: Should say "gpt-5-nano"
   - If "gpt-5" or "gpt-4o": Much more expensive

#### Want to reduce costs

**Options**:

1. **Use smaller batches** (more efficient):
   ```python
   batch_size: int = 100  # From 50 to 100
   ```

2. **Use Maithili (FREE) via IndicTrans2**:
   - See project docs for setup

3. **Use cached prompts** (if doing multiple runs):
   - Script already does this automatically

---

### 🔴 Performance Issues

#### Script is very slow

**Causes & Solutions**:

1. **Poor internet connection**:
   - Check: `speedtest-cli`
   - Solution: Use better network

2. **WSL2 on /mnt/c/**:
   - Solution: Copy to `~/` (see above)

3. **Large batch size**:
   - Reduce to 30-40 for faster responses
   - Trade-off: More API calls

#### Frequent disconnections

**Cause**: Network instability

**Solutions**:
```bash
# Add timeout retry (edit around line ~210):
from requests.exceptions import Timeout

try:
    response = self.client.chat.completions.create(
        model=self.model,
        messages=[...],
        timeout=120  # 2 minutes
    )
except Timeout:
    time.sleep(10)
    # Retry logic here
```

---

### 🔴 Output Issues

#### Output files are empty

**Cause**: Input files might be empty or wrong encoding

**Solutions**:
```bash
# Check input files
wc -l hindi_corpus/*/*.txt

# Check encoding
file hindi_corpus/news/*.txt

# Convert if needed
iconv -f ISO-8859-1 -t UTF-8 input.txt > output.txt
```

#### Directory structure not preserved

**Cause**: Script bug (unlikely) or wrong input

**Solution**:
```bash
# Check your input structure
tree hindi_corpus/

# Should have subdirectories (domains)
# If not, create them:
mkdir -p hindi_corpus/domain1
mv hindi_corpus/*.txt hindi_corpus/domain1/
```

#### Output mixed up between files

**Cause**: Async issue (rare)

**Solution**:
```bash
# Re-run with single language
python translate_domain_corpus.py
# When prompted: Bhojpuri

# Then run again for second language
```

---

## 🔍 Diagnostic Commands

### Check everything:

```bash
# Python version
python --version

# OpenAI package
python -c "import openai; print(openai.__version__)"

# API key
echo $OPENAI_API_KEY | cut -c1-10

# File count
find hindi_corpus -name "*.txt" | wc -l

# Sentence count
find hindi_corpus -name "*.txt" -exec wc -l {} + | tail -1

# Disk space
df -h .

# Network
ping -c 3 api.openai.com
```

### Test translation:

```bash
# Create test file
mkdir -p test_corpus/test
echo "यह एक परीक्षण है।" > test_corpus/test/sample.txt

# Run
python translate_domain_corpus.py
# Enter: test_corpus

# Check output
cat translations/bhojpuri/test/sample.txt
```

---

## 📞 Getting Help

If issue persists:

1. **Check logs**: Look at console output for specific errors
2. **Check API status**: https://status.openai.com
3. **Check quota**: https://platform.openai.com/account/usage
4. **Try test file**: See "Test translation" above
5. **Read README**: Full documentation in `README.md`

---

## 🔧 Advanced Troubleshooting

### Enable debug mode:

Add to script (after imports):
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Save API responses:

Add to script (in translate_batch):
```python
# After response line ~220:
with open(f"debug_batch_{batch_num}.json", "w") as f:
    json.dump(response.model_dump(), f, indent=2, ensure_ascii=False)
```

### Check token usage:

Add to script (after response):
```python
print(f"Tokens: {response.usage.total_tokens}")
```

---

## ✅ Prevention Checklist

Before running:

- [ ] Conda environment activated
- [ ] `pip install openai` completed
- [ ] API key set and verified
- [ ] Input directory exists and has .txt files
- [ ] Files are UTF-8 encoded
- [ ] Disk space available (check `df -h`)
- [ ] Internet connection stable
- [ ] OpenAI account has credits

---

**Version**: 1.0  
**Last Updated**: November 2025  

For more help, see `README.md` or create GitHub issue.
