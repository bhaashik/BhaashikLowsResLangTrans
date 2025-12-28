# 🔐 API Keys Setup Guide

## ⚠️ SECURITY FIRST

**NEVER commit API keys to git!** This guide shows you how to safely store your API keys locally.

## Quick Setup

### 1. Create Local `.env` File

Copy the example template:
```bash
cp .env.example .env
```

### 2. Add Your Real API Keys

Edit `.env` with your actual keys:
```bash
# API Keys - KEEP THIS FILE LOCAL, NEVER COMMIT!

# OpenAI API (GPT-4o, GPT-4o-mini, o1, o3 models)
OPENAI_API_KEY=sk-proj-YOUR_ACTUAL_OPENAI_KEY_HERE

# Anthropic API (Claude Haiku, Sonnet, Opus models)
ANTHROPIC_API_KEY=sk-ant-api03-YOUR_ACTUAL_ANTHROPIC_KEY_HERE

# Google Gemini API (Gemini 1.5 Flash, Pro, 2.0 models)
GOOGLE_API_KEY=YOUR_ACTUAL_GOOGLE_KEY_HERE

# Google Vertex AI (Enterprise Gemini on GCP - optional)
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
```

### 3. Verify `.env` is Gitignored

Check that `.env` won't be committed:
```bash
git status .env
# Should show: "nothing to commit" or not list .env
```

## How the Code Loads Keys

All providers automatically load from environment:

```python
# OpenAI Provider
self.api_key = api_key or os.getenv('OPENAI_API_KEY')

# Anthropic Provider
self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')

# Gemini Provider
self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
```

## Usage Examples

### Option 1: Environment Variables (Recommended)

```bash
# Set in .env file, then:
python scripts/translate_cli.py \
    --provider openai \
    --source-lang hi \
    --target-lang mwr \
    --text "आपका स्वागत है"
```

### Option 2: Pass Directly (Not Recommended for Scripts)

```python
from universal_translate.providers import OpenAIProvider

translator = OpenAIProvider(
    model='gpt-4o-mini',
    api_key='YOUR_KEY_HERE'  # Better to use env var!
)
```

## Security Checklist

- [x] `.env` is in `.gitignore` ✅
- [x] `.env.example` has no real keys ✅
- [x] Code uses `os.getenv()` ✅
- [x] No hardcoded keys in Python files ✅
- [ ] Create `.env` with YOUR real keys (locally only!)
- [ ] NEVER share or commit `.env`

## Getting API Keys

### OpenAI
1. Visit: https://platform.openai.com/api-keys
2. Create new secret key
3. Copy key (starts with `sk-proj-...`)
4. Add to `.env`: `OPENAI_API_KEY=sk-proj-...`

### Anthropic (Claude)
1. Visit: https://console.anthropic.com/settings/keys
2. Create new API key
3. Copy key (starts with `sk-ant-api03-...`)
4. Add to `.env`: `ANTHROPIC_API_KEY=sk-ant-api03-...`

### Google Gemini (Consumer API)
1. Visit: https://aistudio.google.com/app/apikey
2. Create API key
3. Copy key
4. Add to `.env`: `GOOGLE_API_KEY=...`

### Google Vertex AI (Enterprise GCP)
1. Create a GCP project at: https://console.cloud.google.com/
2. Enable Vertex AI API
3. Create a service account with Vertex AI permissions
4. Download service account JSON key
5. Add to `.env`:
   ```
   GOOGLE_CLOUD_PROJECT=your-project-id
   GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
   ```

## What If I Accidentally Commit a Key?

If you accidentally commit a real API key:

1. **Immediately revoke/delete the key** in the provider's console
2. **Remove from git history:**
   ```bash
   git filter-repo --path .env --invert-paths --force
   git push origin main --force
   ```
3. **Create new API key** and add to `.env` (not git!)

## Verification

Test that keys are loading correctly:

```bash
python -c "import os; print('OpenAI:', 'SET' if os.getenv('OPENAI_API_KEY') else 'NOT SET')"
python -c "import os; print('Anthropic:', 'SET' if os.getenv('ANTHROPIC_API_KEY') else 'NOT SET')"
```

---

**Remember:** Your `.env` file stays on YOUR machine only. Never share, commit, or upload it!
