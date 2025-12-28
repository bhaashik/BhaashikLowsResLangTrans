# Translation Providers Guide

This guide describes all available LLM providers for translation in BhaashikLowsResLangTrans.

## Quick Start

```bash
# Set up API keys in .env file (see API_KEYS_SETUP.md)
cp .env.example .env
# Edit .env with your actual API keys

# Translate with default provider (auto-detected)
python scripts/translate_cli.py \
    --source-lang hi \
    --target-lang mwr \
    --text "नमस्ते, आप कैसे हैं?"

# Translate with specific provider and model
python scripts/translate_cli.py \
    --provider anthropic \
    --model claude-haiku-4.5 \
    --source-lang hi \
    --target-lang mwr \
    --text "नमस्ते, आप कैसे हैं?"
```

## Available Providers

### 1. OpenAI (GPT Models)

**Provider name:** `openai`
**Environment variable:** `OPENAI_API_KEY`
**Default model:** `gpt-4o-mini`

#### Available Models

| Model | Input Cost (USD/1M tokens) | Output Cost | Context | Best For |
|-------|---------------------------|-------------|---------|----------|
| `gpt-4o` | $2.50 | $10.00 | 128K | High quality, balanced |
| `gpt-4o-mini` | $0.15 | $0.60 | 128K | **Cost-effective** ⭐ |
| `o1` | $15.00 | $60.00 | 200K | Complex reasoning |
| `o1-mini` | $3.00 | $12.00 | 128K | Affordable reasoning |
| `o3-mini` | $1.10 | $4.40 | 200K | Latest reasoning |
| `gpt-4-turbo` | $10.00 | $30.00 | 128K | High performance |
| `gpt-3.5-turbo` | $0.50 | $1.50 | 16K | Legacy, fast |

**Caching:** 50% discount on cached input tokens
**Batch API:** Not yet integrated
**Rate Limits:** 500 req/min, 150K tokens/min

#### Usage Example

```bash
# Cost-effective translation
python scripts/translate_cli.py \
    --provider openai \
    --model gpt-4o-mini \
    --source-lang hi --target-lang mwr \
    --input input.txt --output output.txt

# High-quality reasoning
python scripts/translate_cli.py \
    --provider openai \
    --model o1-mini \
    --source-lang hi --target-lang bra \
    --text "यह एक जटिल वाक्य है।"
```

---

### 2. Anthropic (Claude Models)

**Provider name:** `anthropic`
**Environment variable:** `ANTHROPIC_API_KEY`
**Default model:** `claude-haiku-4.5`

#### Available Models

| Model | Input Cost (₹/1M tokens) | Output Cost | Context | Best For |
|-------|--------------------------|-------------|---------|----------|
| `claude-haiku-3` | ₹0.21 | ₹1.06 | 200K | Ultra-fast, no caching |
| `claude-haiku-3.5` | ₹0.68 | ₹3.40 | 200K | Fast with caching |
| `claude-haiku-4.5` | ₹0.85 | ₹4.25 | 200K | **Best value** ⭐ |
| `claude-sonnet-3.5` | ₹2.55 | ₹12.75 | 200K | Balanced quality |
| `claude-sonnet-4` | ₹2.55 | ₹12.75 | 200K | Latest Sonnet |
| `claude-sonnet-4.5` | ₹2.55 | ₹12.75 | 200K | Highest Sonnet quality |
| `claude-opus-3` | ₹12.75 | ₹63.75 | 200K | Maximum quality |
| `claude-opus-4` | ₹12.75 | ₹63.75 | 200K | Latest Opus |

**Caching:** 90% discount on cached reads, cache write costs included
**Batch API:** 50% discount on output tokens (use `use_batch_api=True`)
**Rate Limits:** 1000 req/min, 100K tokens/min

#### Usage Example

```bash
# Cost-effective with caching
python scripts/translate_cli.py \
    --provider anthropic \
    --model claude-haiku-4.5 \
    --prompt-config universal_translate/config/prompts/hi_to_mwr_claude.yaml \
    --input input.txt --output output.txt

# High quality
python scripts/translate_cli.py \
    --provider anthropic \
    --model claude-sonnet-4.5 \
    --source-lang hi --target-lang awa \
    --text "यह एक सांस्कृतिक संदर्भ है।"
```

---

### 3. Google Gemini (Consumer API)

**Provider name:** `gemini`
**Environment variable:** `GOOGLE_API_KEY`
**Default model:** `gemini-2.0-flash-exp` (Free!)

#### Available Models

| Model | Input Cost (₹/1M tokens) | Output Cost | Context | Best For |
|-------|--------------------------|-------------|---------|----------|
| `gemini-2.0-flash-exp` | **₹0.00** | **₹0.00** | 1M | **Free tier!** ⭐ |
| `gemini-2.0-flash` | ₹3.19 | ₹9.56 | 1M | Latest, production |
| `gemini-1.5-flash` | ₹3.19 | ₹9.56 | 1M | Fast, cost-effective |
| `gemini-1.5-flash-8b` | ₹1.59 | ₹4.78 | 1M | Smallest, fastest |
| `gemini-1.5-pro` | ₹106.25 | ₹425.00 | 2M | Highest quality |
| `gemini-1.0-pro` | ₹42.50 | ₹127.50 | 32K | Legacy |
| `gemini-pro` | ₹42.50 | ₹127.50 | 32K | Alias for 1.0-pro |

**Caching:** 75% discount on cached input tokens
**Rate Limits:** 60 req/min, 1M tokens/min
**Note:** Free tier has generous quotas, perfect for experimentation!

#### Usage Example

```bash
# Free translation!
python scripts/translate_cli.py \
    --provider gemini \
    --model gemini-2.0-flash-exp \
    --source-lang hi --target-lang mai \
    --input input.txt --output output.txt

# Production with caching
python scripts/translate_cli.py \
    --provider gemini \
    --model gemini-1.5-flash \
    --prompt-config universal_translate/config/prompts/hi_to_brx_openai.yaml \
    --input large_file.txt --output output.txt
```

---

### 4. Google Vertex AI (Enterprise GCP)

**Provider name:** `vertex`
**Environment variables:** `GOOGLE_CLOUD_PROJECT`, `GOOGLE_APPLICATION_CREDENTIALS`
**Default model:** `gemini-1.5-flash-002`

#### Available Models

| Model | Input Cost (₹/1M tokens) | Output Cost | Context | Best For |
|-------|--------------------------|-------------|---------|----------|
| `gemini-2.0-flash-exp` | **₹0.00** | **₹0.00** | 1M | Free experimental |
| `gemini-1.5-flash-002` | ₹1.59 | ₹6.37 | 1M | **Enterprise tier** ⭐ |
| `gemini-1.5-pro-002` | ₹53.13 | ₹212.50 | 2M | Enterprise quality |
| `gemini-1.0-pro` | ₹21.25 | ₹63.75 | 32K | Legacy enterprise |

**Caching:** 75% discount on cached input tokens
**Rate Limits:** 300 req/min, 1M tokens/min
**Benefits:** Enterprise SLA, VPC-SC, CMEK, audit logs, regional deployment

**Note:** Vertex AI pricing is ~50% cheaper than Gemini API for the same models!

#### Setup

1. Create GCP project
2. Enable Vertex AI API
3. Create service account with `Vertex AI User` role
4. Download service account JSON key
5. Set environment variables:
   ```bash
   GOOGLE_CLOUD_PROJECT=your-project-id
   GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
   ```

#### Usage Example

```bash
# Enterprise translation with SLA
python scripts/translate_cli.py \
    --provider vertex \
    --model gemini-1.5-flash-002 \
    --source-lang hi --target-lang mwr \
    --input input.txt --output output.txt
```

---

## Provider Comparison

### Cost Comparison (for 1M input + 1M output tokens)

| Provider | Cheapest Option | Cost (₹) | Cost (USD) | Notes |
|----------|----------------|----------|------------|-------|
| Gemini | gemini-2.0-flash-exp | **₹0.00** | **$0.00** | Free tier! |
| Vertex AI | gemini-1.5-flash-002 | ₹7.96 | $0.09 | Enterprise SLA |
| OpenAI | gpt-4o-mini | ₹63.75 | $0.75 | With 50% caching |
| Anthropic | claude-haiku-4.5 | ₹5.10 | $0.06 | **Best paid value** |

### Quality vs Cost (Ranking)

#### Budget Tier (< ₹10/1M tokens)
1. 🥇 `claude-haiku-4.5` - Best quality for price
2. 🥈 `gemini-1.5-flash-8b` - Fastest, smallest
3. 🥉 `gemini-2.0-flash-exp` - Free!

#### Balanced Tier (₹10-100/1M tokens)
1. 🥇 `claude-sonnet-4.5` - Best overall
2. 🥈 `gpt-4o` - Strong reasoning
3. 🥉 `o3-mini` - Latest reasoning

#### Premium Tier (> ₹100/1M tokens)
1. 🥇 `claude-opus-4` - Maximum quality
2. 🥈 `o1` - Best reasoning
3. 🥉 `gemini-1.5-pro` - Long context

---

## Using Prompt Caching

All providers support prompt caching to reduce costs on repeated translations.

### Example with Caching

```bash
# First run: Full cost, creates cache
python scripts/translate_cli.py \
    --provider anthropic \
    --model claude-haiku-4.5 \
    --prompt-config universal_translate/config/prompts/hi_to_mwr_claude.yaml \
    --input batch1.txt

# Subsequent runs: 90% savings on cached prompt!
python scripts/translate_cli.py \
    --provider anthropic \
    --model claude-haiku-4.5 \
    --prompt-config universal_translate/config/prompts/hi_to_mwr_claude.yaml \
    --input batch2.txt
```

**Caching Benefits:**
- **Anthropic:** 90% discount on cached reads (₹0.085 vs ₹0.85)
- **OpenAI:** 50% discount on cached input ($0.075 vs $0.15)
- **Gemini/Vertex:** 75% discount on cached input (₹0.40 vs ₹1.59)

---

## Recommended Strategies

### For Development/Testing
```bash
# Use free Gemini tier
--provider gemini --model gemini-2.0-flash-exp
```

### For Production (Cost-Optimized)
```bash
# Claude Haiku with prompt caching
--provider anthropic --model claude-haiku-4.5 \
--prompt-config universal_translate/config/prompts/hi_to_mwr_claude.yaml
```

### For Production (Quality-Optimized)
```bash
# Claude Sonnet for best balance
--provider anthropic --model claude-sonnet-4.5 \
--prompt-config universal_translate/config/prompts/hi_to_mwr_claude.yaml
```

### For Maximum Quality
```bash
# Claude Opus or o1
--provider anthropic --model claude-opus-4
# OR
--provider openai --model o1
```

### For Large-Scale (Enterprise)
```bash
# Vertex AI with enterprise SLA
--provider vertex --model gemini-1.5-flash-002
```

---

## Environment Setup

Create `.env` file with your API keys (never commit this file!):

```bash
# OpenAI
OPENAI_API_KEY=sk-proj-...

# Anthropic
ANTHROPIC_API_KEY=sk-ant-api03-...

# Google Gemini (Consumer)
GOOGLE_API_KEY=AIza...

# Google Vertex AI (Enterprise)
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

See `API_KEYS_SETUP.md` for detailed setup instructions.

---

## Troubleshooting

### "API key not found"
- Ensure `.env` file exists and contains the correct key
- Check environment variable name matches provider requirement
- For Vertex AI, ensure both project ID and credentials path are set

### "Rate limit exceeded"
- Reduce `--batch-size` parameter
- Add delays between requests
- Upgrade to higher tier (contact provider)

### "Model not found"
- Check model name spelling: `--list-models <provider>`
- Some models require API access approval
- Vertex AI models have region-specific availability

### "Authentication failed" (Vertex AI)
- Verify service account has `Vertex AI User` role
- Check credentials file path is correct and accessible
- Ensure Vertex AI API is enabled in GCP project

---

## Cost Optimization Tips

1. **Use Free Tier First:** Start with `gemini-2.0-flash-exp` for development
2. **Enable Prompt Caching:** Use `--prompt-config` to cache examples (90% savings!)
3. **Batch Processing:** Process multiple texts together to amortize prompt costs
4. **Choose Right Model:** Don't use Opus/o1 for simple translations
5. **Monitor Usage:** Use `--show-cost` flag to track spending
6. **Vertex vs Gemini:** Use Vertex AI for ~50% cost savings on production workloads

---

## API Documentation Links

- **OpenAI:** https://platform.openai.com/docs/
- **Anthropic:** https://docs.anthropic.com/claude/docs
- **Google Gemini:** https://ai.google.dev/docs
- **Google Vertex AI:** https://cloud.google.com/vertex-ai/docs

---

**Last Updated:** December 2024
