# LLM-Agnostic Translation Framework

Complete guide to using the unified multi-provider translation framework.

## Overview

This framework provides a plugin-based architecture for translating low-resource Indic languages using multiple LLM providers (OpenAI, Anthropic, Gemini) through a single unified interface.

### Key Features

- **Provider Agnostic**: Switch between OpenAI, Anthropic, and Gemini with a single CLI argument
- **Automatic Prompt Caching**: Cost optimization with provider-specific caching (50% OpenAI, 90% Anthropic, 75% Gemini)
- **Unified CLI Tool**: Single command-line interface for all providers
- **Cost Tracking**: Automatic cost calculation and reporting
- **Batch Processing**: Efficient batch translation with caching benefits
- **Python API**: Programmatic access through simple API

## Supported Providers

| Provider | Models | Caching | Default Model |
|----------|--------|---------|---------------|
| **OpenAI** | gpt-4o-mini, gpt-4o, gpt-5-nano | 50% discount | gpt-4o-mini |
| **Anthropic** | claude-haiku-4.5, claude-sonnet-4.5, claude-haiku-3.5 | 90% discount | claude-haiku-4.5 |
| **Gemini** | gemini-2.0-flash-exp (free), gemini-1.5-flash, gemini-1.5-pro | 75% discount | gemini-1.5-flash |

## Supported Languages

- **Awadhi** (awa)
- **Braj** (bra)
- **Marwari** (mwr)
- **Maithili** (mai)
- **Nepali** (nep)
- **Bhojpuri** (bho)
- **Magahi** (mag)

## Quick Start

### 1. Set up API Keys

```bash
# OpenAI
export OPENAI_API_KEY="your-openai-key"

# Anthropic (Claude)
export ANTHROPIC_API_KEY="your-anthropic-key"

# Google (Gemini)
export GOOGLE_API_KEY="your-google-key"
```

### 2. List Available Providers

```bash
python scripts/translate_cli.py --list-providers
```

Output:
```
Available providers:
  [✓] openai       (default: gpt-4o-mini, env: OPENAI_API_KEY)
  [✗] anthropic    (default: claude-haiku-4.5, env: ANTHROPIC_API_KEY)
  [✓] gemini       (default: gemini-1.5-flash, env: GOOGLE_API_KEY)
```

### 3. Translate with Different Providers

```bash
# OpenAI GPT-4o-mini (most accessible, good quality)
python scripts/translate_cli.py \
    --provider openai \
    --source-lang hi \
    --target-lang mwr \
    --text "मैं आज बाजार जा रहा हूं।"

# Anthropic Claude (best quality, highest savings)
python scripts/translate_cli.py \
    --provider anthropic \
    --model claude-haiku-4.5 \
    --source-lang hi \
    --target-lang bra \
    --text "यह फूल बहुत सुंदर है।"

# Google Gemini (free tier available)
python scripts/translate_cli.py \
    --provider gemini \
    --model gemini-2.0-flash-exp \
    --source-lang hi \
    --target-lang awa \
    --text "वह स्कूल जा रहा है।"
```

## CLI Usage

### Basic Translation

```bash
# Translate a single sentence
python scripts/translate_cli.py \
    --provider openai \
    --source-lang hi \
    --target-lang mwr \
    --text "मुझे किताबें पढ़ना पसंद है।"

# Translate from file
python scripts/translate_cli.py \
    --provider anthropic \
    --source-lang hi \
    --target-lang bra \
    --input input.txt \
    --output output.txt
```

### Using Prompt Configurations

```bash
# Use provider-specific prompt config
python scripts/translate_cli.py \
    --provider openai \
    --prompt-config universal_translate/config/prompts/hi_to_mwr_openai.yaml \
    --input input.txt \
    --output output.txt

# Framework auto-detects configs if available
python scripts/translate_cli.py \
    --provider anthropic \
    --source-lang hi \
    --target-lang mai \
    --input input.txt  # Will use hi_to_mai_claude.yaml if exists
```

### Batch Processing with Cost Tracking

```bash
# Translate file with cost summary
python scripts/translate_cli.py \
    --provider openai \
    --model gpt-4o-mini \
    --source-lang hi \
    --target-lang mwr \
    --input hindi_sentences.txt \
    --output marwari_translations.txt \
    --batch-size 20 \
    --show-cost
```

Output:
```
Creating openai translator...
Model: gpt-4o-mini (default)
Translator initialized successfully.
Loaded 100 sentences from hindi_sentences.txt
Translating 100 sentence(s)...

============================================================
COST SUMMARY
============================================================
Provider: openai
Model: gpt-4o-mini
Total cost: ₹3.8542 INR ($0.0453 USD)
Success rate: 100/100 (100.0%)
Average cost per sentence: ₹0.038542 ($0.000453 USD)
Cache hits: 99/100
============================================================
```

### Advanced Options

```bash
# Custom temperature and max tokens
python scripts/translate_cli.py \
    --provider gemini \
    --source-lang hi \
    --target-lang nep \
    --input input.txt \
    --temperature 0.1 \
    --max-tokens 4096

# Specify API key directly
python scripts/translate_cli.py \
    --provider openai \
    --api-key sk-... \
    --source-lang hi \
    --target-lang awa \
    --text "आज मौसम बहुत अच्छा है।"

# Quiet mode (only output translations)
python scripts/translate_cli.py \
    --provider openai \
    --source-lang hi \
    --target-lang mwr \
    --input input.txt \
    --quiet > output.txt
```

## Python API Usage

### Simple Translation

```python
from universal_translate.providers import create_translator
from universal_translate.core import TranslationUnit, TranslationRequest

# Create translator (auto-detects prompt config)
translator = create_translator(
    provider='openai',           # or 'anthropic', 'gemini'
    model='gpt-4o-mini',         # optional, uses default if not specified
    source_lang='hi',
    target_lang='mwr'
)

# Initialize
translator.initialize()

# Create translation request
units = [TranslationUnit(text="मैं घर जा रहा हूं।", index=0)]
request = TranslationRequest(
    units=units,
    src_lang='hi',
    tgt_lang='mwr',
    parameters={'temperature': 0.3, 'max_tokens': 2048}
)

# Translate
response = translator.translate_sync(request)

# Get translation
translation = response.results[0].translation
print(f"Translation: {translation}")
print(f"Cost: ₹{response.total_cost:.4f} INR")
```

### Using Provider Registry

```python
from universal_translate.providers import ProviderRegistry

# List all providers
providers = ProviderRegistry.list_providers()
print(f"Available: {providers}")

# Check which providers are configured
available = ProviderRegistry.list_available_providers()
print(f"Configured: {available}")

# Get provider info
info = ProviderRegistry.get_provider_info('openai')
print(f"Models: {info['models']}")
print(f"Default: {info['default_model']}")

# Create provider with custom config
translator = ProviderRegistry.create_provider(
    provider='anthropic',
    model='claude-haiku-4.5',
    prompt_config='universal_translate/config/prompts/hi_to_bra_claude.yaml'
)
```

### Batch Translation

```python
from pathlib import Path
from universal_translate.providers import create_translator
from universal_translate.core import TranslationUnit, TranslationRequest

# Load sentences
sentences = Path('input.txt').read_text().splitlines()

# Create translator
translator = create_translator(
    provider='openai',
    source_lang='hi',
    target_lang='mwr'
)
translator.initialize()

# Create batch request
units = [
    TranslationUnit(text=text, index=i)
    for i, text in enumerate(sentences)
]

request = TranslationRequest(
    units=units,
    src_lang='hi',
    tgt_lang='mwr'
)

# Translate
response = translator.translate_sync(request)

# Extract translations
translations = [r.translation for r in response.results]

# Save output
Path('output.txt').write_text('\n'.join(translations))

print(f"Translated {len(translations)} sentences")
print(f"Total cost: ₹{response.total_cost:.4f} INR")
```

### Switching Between Providers

```python
from universal_translate.providers import create_translator

# Compare translations from different providers
providers_config = [
    {'provider': 'openai', 'model': 'gpt-4o-mini'},
    {'provider': 'anthropic', 'model': 'claude-haiku-4.5'},
    {'provider': 'gemini', 'model': 'gemini-1.5-flash'}
]

text = "मैं आज बाजार जा रहा हूं।"

for config in providers_config:
    translator = create_translator(
        provider=config['provider'],
        model=config['model'],
        source_lang='hi',
        target_lang='mwr'
    )
    translator.initialize()

    # Translate
    from universal_translate.core import TranslationUnit, TranslationRequest
    request = TranslationRequest(
        units=[TranslationUnit(text=text, index=0)],
        src_lang='hi',
        tgt_lang='mwr'
    )

    response = translator.translate_sync(request)
    translation = response.results[0].translation

    print(f"{config['provider']:12s} | {translation}")
    print(f"             | Cost: ₹{response.total_cost:.6f}")
```

## Cost Comparison

### Example: Translating 1,000 Hindi Sentences to Marwari

| Provider | Model | Total Cost (INR) | Cost per Sentence | Caching Savings |
|----------|-------|------------------|-------------------|-----------------|
| OpenAI | gpt-4o-mini | ₹38.30 | ₹0.0383 | 45% (₹31.27 saved) |
| Anthropic | claude-haiku-4.5 | ₹42.60 | ₹0.0426 | 96% (₹1,020 saved) |
| Gemini | gemini-1.5-flash | ₹31.90 | ₹0.0319 | 75% (₹95.70 saved) |
| Gemini | gemini-2.0-flash-exp | ₹0.00 | ₹0.0000 | Free tier |

**Note**: Costs assume ~100 examples in prompt cache (reused for all sentences after first).

## Prompt Configuration

### Directory Structure

```
universal_translate/config/prompts/
├── hi_to_awa_openai.yaml      # OpenAI config for Awadhi
├── hi_to_awa_claude.yaml      # Claude config for Awadhi
├── hi_to_bra_openai.yaml      # OpenAI config for Braj
├── hi_to_bra_claude.yaml      # Claude config for Braj
├── hi_to_mwr_openai.yaml      # OpenAI config for Marwari
├── hi_to_mwr_claude.yaml      # Claude config for Marwari
├── hi_to_mai_openai.yaml      # OpenAI config for Maithili
├── hi_to_mai_claude.yaml      # Claude config for Maithili
├── hi_to_nep_openai.yaml      # OpenAI config for Nepali
└── hi_to_nep_claude.yaml      # Claude config for Nepali
```

### Creating Provider-Agnostic Prompts

The framework supports both provider-specific and generic prompt configs. Provider-specific configs are automatically selected:

```bash
# Looks for: hi_to_mwr_openai.yaml first, then hi_to_mwr.yaml
python scripts/translate_cli.py --provider openai --source-lang hi --target-lang mwr --input file.txt

# Looks for: hi_to_mwr_claude.yaml first, then hi_to_mwr.yaml
python scripts/translate_cli.py --provider anthropic --source-lang hi --target-lang mwr --input file.txt
```

## Best Practices

### 1. Choose the Right Provider

- **OpenAI (gpt-4o-mini)**: Best for accessibility and good value
- **Anthropic (claude-haiku-4.5)**: Best for quality and maximum caching savings
- **Gemini (gemini-2.0-flash-exp)**: Best for free tier / experimentation

### 2. Optimize for Caching

```python
# Use all available examples for maximum cache benefit
from universal_translate.prompts import PromptManager

pm = PromptManager('universal_translate/config/prompts/hi_to_mwr_openai.yaml')
examples = pm.format_examples_for_prompt(max_examples=None)  # Use all
```

### 3. Batch Processing

Process sentences in batches within cache TTL (5-10 minutes):

```bash
# Process in one session to maximize caching
python scripts/translate_cli.py \
    --provider openai \
    --source-lang hi --target-lang mwr \
    --input all_sentences.txt \
    --batch-size 20
```

### 4. Cost Monitoring

Always use `--show-cost` during development:

```bash
python scripts/translate_cli.py \
    --provider anthropic \
    --source-lang hi --target-lang bra \
    --input test.txt \
    --show-cost
```

## Troubleshooting

### Provider Not Available

```bash
$ python scripts/translate_cli.py --provider anthropic ...
Error: API key required for anthropic. Set ANTHROPIC_API_KEY environment variable
```

**Solution**: Set the appropriate API key:
```bash
export ANTHROPIC_API_KEY="your-key"
```

### Prompt Config Not Found

If auto-detection fails, specify explicitly:

```bash
python scripts/translate_cli.py \
    --provider openai \
    --prompt-config universal_translate/config/prompts/hi_to_mwr_openai.yaml \
    --input file.txt
```

### Low Cache Hit Rate

Check that:
1. System prompt is >1024 tokens (OpenAI requirement)
2. Requests happen within cache TTL (5-10 minutes)
3. Examples are loaded correctly

### Installation Issues

```bash
# OpenAI
pip install openai

# Anthropic
pip install anthropic

# Gemini
pip install google-generativeai
```

## Architecture

### Provider Plugin System

```
ProviderRegistry
    ├── OpenAIProvider (extends BaseTranslator)
    ├── AnthropicProvider (extends BaseTranslator)
    └── GeminiProvider (extends BaseTranslator)
```

Each provider implements:
- `translate()`: Async translation with caching
- `translate_sync()`: Synchronous wrapper
- `get_cost_estimate()`: Cost estimation
- `supports_prompt_caching()`: Caching capability

### CLI Flow

```
CLI Input → ProviderRegistry → Provider → LLM API → Results → CLI Output
                              ↓
                        PromptManager
                              ↓
                      Load Examples & Config
```

## References

- [OpenAI Prompt Caching](https://platform.openai.com/docs/guides/prompt-caching)
- [Anthropic Prompt Caching](https://docs.anthropic.com/claude/docs/prompt-caching)
- [Gemini Context Caching](https://ai.google.dev/docs/caching)
- [OpenAI Guide](OPENAI_PROMPT_CACHING_GUIDE.md)
- [Anthropic Guide](PROMPT_CACHING_GUIDE.md)
- [API Comparison](API_COMPARISON.md)
- [Data Sources](DATA_SOURCES.md)
