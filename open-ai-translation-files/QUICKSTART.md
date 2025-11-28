# QUICK START - Hindi Translation (2 Minutes)

## For WSL2 Ubuntu Users (FASTEST):

```bash
# 1. Run setup script
bash setup_wsl2.sh

# 2. Follow the prompts - done!
```

---

## For Manual Setup:

```bash
# 1. Install OpenAI package
pip install openai

# 2. Set API key (get from: https://platform.openai.com/api-keys)
export OPENAI_API_KEY="sk-proj-your-key-here"

# 3. Prepare your corpus
mkdir -p hindi_corpus/domain1
echo "आपका हिंदी वाक्य यहां" > hindi_corpus/domain1/file1.txt

# 4. Run translation
python translate_domain_corpus.py
```

---

## What It Does:

- Translates Hindi → Bhojpuri & Magahi
- Preserves your directory structure
- Excludes temp files (.swp, .tmp, .bak)
- Ultra-fast with GPT-5 Nano
- Ultra-cheap: ~₹91 per language for 35K sentences

---

## Example Cost:

| Sentences | Both Languages |
|-----------|----------------|
| 10,000 | ₹52 ($0.60) |
| 35,000 | ₹182 ($2.14) |
| 70,000 | ₹366 ($4.30) |

---

## Need Help?

Read the full `README.md` for:
- Detailed setup instructions
- Troubleshooting guide
- Advanced features
- Linguistic information

---

## File Structure:

```
Your Setup/
├── translate_domain_corpus.py   ← Main script
├── setup_wsl2.sh                ← Auto setup (WSL2)
├── requirements.txt             ← Dependencies
├── README.md                    ← Full guide
└── QUICKSTART.md               ← This file
```

---

## Your Corpus Structure:

```
hindi_corpus/
├── news/
│   └── file1.txt     ← One sentence per line
├── literature/
│   └── file2.txt
└── social/
    └── file3.txt
```

---

## After Translation:

```
translations/
├── bhojpuri/
│   ├── news/
│   ├── literature/
│   └── social/
└── magahi/
    └── (same structure)
```

---

**That's it! Ready to translate! 🚀**

For complete documentation, see: `README.md`
