# Quick Reference: Translation Providers

## Setup (One-Time)

```bash
# 1. Copy environment template
cp .env.example .env

# 2. Add your API keys to .env (see API_KEYS_SETUP.md)
# Edit .env with your keys - NEVER commit this file!

# 3. Test providers
python scripts/test_all_providers.py --list-available
```

## Basic Usage

```bash
# Translate a single text (auto-detects provider)
python scripts/translate_cli.py \
    --source-lang hi --target-lang mwr \
    --text "नमस्ते, आप कैसे हैं?"

# Translate a file
python scripts/translate_cli.py \
    --provider anthropic --model claude-haiku-4.5 \
    --source-lang hi --target-lang mwr \
    --input input.txt --output output.txt
```

## Provider Quick Reference

### 🆓 Free Option
```bash
--provider gemini --model gemini-2.0-flash-exp
```

### 💰 Best Value (Paid)
```bash
--provider anthropic --model claude-haiku-4.5
```

### ⚡ Fastest (Cheap)
```bash
--provider gemini --model gemini-1.5-flash-8b
```

### 🎯 Best Quality
```bash
--provider anthropic --model claude-sonnet-4.5
```

### 🏢 Enterprise (GCP)
```bash
--provider vertex --model gemini-1.5-flash-002
```

## All Available Models

### OpenAI
```bash
gpt-4o-mini      # $0.15/$0.60 - Best balance ⭐
gpt-4o           # $2.50/$10.00 - High quality
o1-mini          # $3.00/$12.00 - Reasoning
o3-mini          # $1.10/$4.40 - Latest reasoning
o1               # $15.00/$60.00 - Best reasoning
gpt-4-turbo      # $10.00/$30.00 - Fast
```

### Anthropic
```bash
claude-haiku-4.5    # ₹0.85/₹4.25 - Best value ⭐
claude-haiku-3.5    # ₹0.68/₹3.40 - Fast with caching
claude-sonnet-4.5   # ₹2.55/₹12.75 - Best quality
claude-sonnet-4     # ₹2.55/₹12.75 - Latest Sonnet
claude-opus-4       # ₹12.75/₹63.75 - Maximum quality
```

### Google Gemini
```bash
gemini-2.0-flash-exp   # FREE - Experimental ⭐
gemini-1.5-flash-8b    # ₹1.59/₹4.78 - Fastest
gemini-1.5-flash       # ₹3.19/₹9.56 - Balanced
gemini-2.0-flash       # ₹3.19/₹9.56 - Latest
gemini-1.5-pro         # ₹106.25/₹425 - Quality
```

### Google Vertex AI
```bash
gemini-1.5-flash-002   # ₹1.59/₹6.37 - Enterprise ⭐
gemini-1.5-pro-002     # ₹53.13/₹212.50 - Quality
gemini-2.0-flash-exp   # FREE - Experimental
```

## Cost Comparison (1M in + 1M out)

| Provider | Model | Cost | Use Case |
|----------|-------|------|----------|
| Gemini | 2.0-flash-exp | **FREE** | Testing/Dev |
| Anthropic | haiku-4.5 | ₹5.10 | **Production** ⭐ |
| Vertex | 1.5-flash-002 | ₹7.96 | Enterprise |
| Gemini | 1.5-flash-8b | ₹6.37 | Speed |
| OpenAI | gpt-4o-mini | ₹63.75 | OpenAI users |

## Enable Prompt Caching (90% savings!)

```bash
# Use a prompt config to enable caching
python scripts/translate_cli.py \
    --provider anthropic \
    --prompt-config universal_translate/config/prompts/hi_to_mwr_claude.yaml \
    --input large_file.txt
```

Available prompt configs:
- `hi_to_mwr_openai.yaml` / `hi_to_mwr_claude.yaml` - Marwari
- `hi_to_awa_openai.yaml` / `hi_to_awa_claude.yaml` - Awadhi
- `hi_to_bra_openai.yaml` / `hi_to_bra_claude.yaml` - Braj
- `hi_to_brx_openai.yaml` / `hi_to_brx_claude.yaml` - Bodo
- `hi_to_mai_openai.yaml` / `hi_to_mai_claude.yaml` - Maithili
- `hi_to_nep_openai.yaml` / `hi_to_nep_claude.yaml` - Nepali

## View Costs

```bash
# Show cost summary after translation
python scripts/translate_cli.py ... --show-cost

# Test all providers and compare costs
python scripts/test_all_providers.py
```

## List Available Options

```bash
# List all providers
python scripts/translate_cli.py --list-providers

# List models for a provider
python scripts/translate_cli.py --list-models openai
python scripts/translate_cli.py --list-models anthropic
python scripts/translate_cli.py --list-models gemini
python scripts/translate_cli.py --list-models vertex
```

## Common Workflows

### Development/Testing
```bash
# Use free Gemini
python scripts/translate_cli.py \
    --provider gemini --model gemini-2.0-flash-exp \
    --source-lang hi --target-lang mwr \
    --input test.txt
```

### Production (Cost-Optimized)
```bash
# Use Claude Haiku with caching
python scripts/translate_cli.py \
    --provider anthropic --model claude-haiku-4.5 \
    --prompt-config universal_translate/config/prompts/hi_to_mwr_claude.yaml \
    --input production.txt --output results.txt \
    --show-cost
```

### Production (Quality-Optimized)
```bash
# Use Claude Sonnet
python scripts/translate_cli.py \
    --provider anthropic --model claude-sonnet-4.5 \
    --prompt-config universal_translate/config/prompts/hi_to_mwr_claude.yaml \
    --input production.txt --output results.txt
```

### Enterprise (GCP)
```bash
# Use Vertex AI with SLA
python scripts/translate_cli.py \
    --provider vertex --model gemini-1.5-flash-002 \
    --source-lang hi --target-lang mwr \
    --input enterprise.txt --output results.txt
```

## Troubleshooting

### No providers available
```bash
# Check which API keys are set
python scripts/test_all_providers.py --list-available

# Set API keys in .env file (see API_KEYS_SETUP.md)
```

### Rate limit exceeded
```bash
# Reduce batch size
python scripts/translate_cli.py --batch-size 5 ...
```

### Vertex AI authentication failed
```bash
# Check environment variables
echo $GOOGLE_CLOUD_PROJECT
echo $GOOGLE_APPLICATION_CREDENTIALS

# Verify credentials file exists
ls -l $GOOGLE_APPLICATION_CREDENTIALS
```

## More Information

- **Full Guide:** `PROVIDERS.md`
- **API Setup:** `API_KEYS_SETUP.md`
- **Project Guide:** `CLAUDE.md`
