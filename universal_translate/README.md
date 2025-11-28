# Universal Translation Package

A flexible, extensible Python framework for machine translation that supports multiple providers, custom prompting strategies, and any language pair.

## Features

- **Multiple Providers**: Anthropic Claude, OpenAI, Google Translate, Azure, DeepL, local models (IndicTrans2, NLLB)
- **External Configuration**: All prompts and examples loaded from YAML/JSON files (no hardcoding)
- **Prompt Caching**: Automatic cost optimization for providers supporting caching (Claude)
- **Cost Tracking**: Real-time cost tracking across all providers
- **Flexible Prompting**: Template-based prompting with variable substitution
- **Example-Based Learning**: Load translation examples from external files
- **Format Agnostic**: Supports text, CoNLL-U, JSON, XML, and custom formats
- **Directory Preservation**: Maintains source directory structure in outputs
- **Async Support**: Asynchronous translation for improved performance

## Installation

```bash
# Clone or copy the universal_translate package
cp -r universal_translate /path/to/your/project/

# Install dependencies
pip install anthropic pyyaml  # For Claude API support
pip install openai            # For OpenAI support
pip install google-cloud-translate  # For Google Translate
```

## Quick Start

### Basic Translation

```python
from universal_translate import (
    AnthropicProvider,
    PromptManager,
    TranslationRequest,
    TranslationUnit
)

# Load prompt configuration
prompt_mgr = PromptManager("universal_translate/config/prompts/low_resource.yaml")

# Create provider
provider = AnthropicProvider(
    model="claude-haiku-4.5",
    prompt_manager=prompt_mgr
)

# Create request
request = TranslationRequest(
    units=[
        TranslationUnit(text="राम घर जा रहे हैं।", index=0),
        TranslationUnit(text="यह किताब अच्छी है।", index=1)
    ],
    src_lang="hi",
    tgt_lang="bho"
)

# Translate
with provider:
    response = provider.translate_sync(request)

# Print results
for result in response.results:
    print(f"Source: {result.source}")
    print(f"Translation: {result.translation}")
    print(f"Cost: ₹{result.cost:.4f}")
    print()

print(f"Total Cost: ₹{response.total_cost:.2f}")
```

## Configuration

### Prompt Configuration (YAML)

Create custom prompt templates in `config/prompts/`:

```yaml
# config/prompts/my_prompt.yaml
name: "my_custom_prompt"
description: "Custom translation prompt"

system_prompt: |
  You are an expert translator for ${source_lang} to ${target_lang}.
  Maintain accuracy and natural fluency.

user_prompt: |
  Translate this ${source_lang} text to ${target_lang}:

  ${text}

# Optional: Path to examples file
examples_file: "../../data/examples/my_examples.json"

# Enable caching (for Claude)
use_prompt_caching: true

# Model parameters
parameters:
  temperature: 0.3
  max_tokens: 2048
```

### Translation Examples (JSON)

Create translation examples in `data/examples/`:

```json
{
  "examples": [
    {
      "source": "Hello, how are you?",
      "target": "नमस्ते, आप कैसे हैं?",
      "source_lang": "en",
      "target_lang": "hi",
      "context": "greeting"
    }
  ],
  "metadata": {
    "created_by": "human_annotator",
    "created_date": "2025-12-01"
  }
}
```

### Provider Configuration

Providers are configured in `config/providers.yaml`. You can customize:
- API keys (via environment variables)
- Model costs
- Rate limits
- Batch API settings

## Architecture

```
universal_translate/
├── core/                   # Core interfaces
│   ├── base_translator.py # Abstract base class
│   └── translation_request.py  # Data models
│
├── providers/              # Provider implementations
│   ├── anthropic_provider.py
│   ├── openai_provider.py
│   └── google_provider.py
│
├── prompts/                # Prompt management
│   └── prompt_manager.py
│
├── config/                 # Configuration files
│   ├── providers.yaml
│   └── prompts/
│       ├── general.yaml
│       └── low_resource.yaml
│
└── data/                   # Translation examples
    └── examples/
        └── hi_to_bho_examples.json
```

## Usage Examples

### With Custom Prompts

```python
from universal_translate import AnthropicProvider, PromptManager

# Load custom prompt
prompt_mgr = PromptManager("path/to/my_prompt.yaml")

# Create provider with prompt
provider = AnthropicProvider(
    model="claude-haiku-4.5",
    prompt_manager=prompt_mgr
)
```

### Cost Estimation

```python
# Estimate cost before translation
estimate = provider.get_cost_estimate(request)

print(f"Estimated cost: ₹{estimate.total_cost:.2f}")
print(f"Input cost: ₹{estimate.input_cost:.2f}")
print(f"Output cost: ₹{estimate.output_cost:.2f}")
print(f"Cost per unit: ₹{estimate.total_cost/estimate.num_units:.4f}")

# Proceed if acceptable
if estimate.total_cost < 100:  # Budget check
    response = provider.translate_sync(request)
```

### Batch Translation

```python
# Create larger batch
units = [
    TranslationUnit(text=text, index=i)
    for i, text in enumerate(text_list)
]

request = TranslationRequest(
    units=units,
    src_lang="hi",
    tgt_lang="bho"
)

# Use batch API for cost savings
provider = AnthropicProvider(
    model="claude-haiku-4.5",
    use_batch_api=True  # 50% off output tokens
)

response = provider.translate_sync(request)
```

### Prompt Caching for Cost Optimization

```python
# Enable caching in prompt config
# config/prompts/cached_prompt.yaml:
#   use_prompt_caching: true

prompt_mgr = PromptManager("config/prompts/cached_prompt.yaml")

provider = AnthropicProvider(
    model="claude-haiku-4.5",  # Must support caching
    prompt_manager=prompt_mgr
)

# First request: pays full cost, creates cache
response1 = provider.translate_sync(request1)

# Subsequent requests: 90% discount on cached prompts!
response2 = provider.translate_sync(request2)
```

## Adding New Providers

1. Create new file in `providers/`:

```python
# providers/my_provider.py
from ..core import BaseTranslator, TranslationRequest, TranslationResponse

class MyProvider(BaseTranslator):
    def __init__(self, api_key: str, **kwargs):
        super().__init__(name='my_provider', **kwargs)
        self.api_key = api_key

    async def translate(self, request: TranslationRequest) -> TranslationResponse:
        # Implement translation logic
        pass

    def translate_sync(self, request: TranslationRequest) -> TranslationResponse:
        # Synchronous wrapper
        pass

    def get_cost_estimate(self, request: TranslationRequest) -> CostEstimate:
        # Calculate cost
        pass

    def supports_batch(self) -> bool:
        return True

    def supports_prompt_caching(self) -> bool:
        return False

    def get_max_tokens(self) -> int:
        return 4096

    def get_rate_limits(self) -> Dict[str, int]:
        return {'requests_per_minute': 100}
```

2. Add to `providers/__init__.py`
3. Add configuration to `config/providers.yaml`

## Environment Variables

Required environment variables per provider:

```bash
# Anthropic Claude
export ANTHROPIC_API_KEY="your_key_here"

# OpenAI
export OPENAI_API_KEY="your_key_here"

# Google Cloud Translate
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"

# Azure
export AZURE_TRANSLATOR_KEY="your_key_here"
export AZURE_TRANSLATOR_REGION="your_region"

# DeepL
export DEEPL_API_KEY="your_key_here"
```

## Design Principles

1. **Provider Abstraction**: All providers implement `BaseTranslator` interface
2. **External Configuration**: No hardcoded prompts or examples
3. **Composability**: Mix and match providers, prompts, and processors
4. **Testability**: Mock providers for testing
5. **Extensibility**: Easy to add new providers and formats

## Comparison with Current Codebase

**Before (current code)**:
- Prompts hardcoded in Python files
- Examples embedded in code
- Provider-specific implementations scattered
- Tight coupling between components

**After (universal_translate)**:
- All prompts in YAML config files
- Examples in JSON data files
- Unified provider interface
- Clear separation of concerns
- Easy to switch between providers

## Performance Optimization

### Prompt Caching (Claude)

For repeated translations with same language pair:
- System prompts cached automatically
- 90% cost reduction on cached content
- Especially effective for large example sets

### Batch API (Claude)

- 50% discount on output tokens
- Ideal for large-scale translation projects

### Rate Limiting

- Automatic rate limiting per provider
- Exponential backoff on errors
- Token-based limiting for optimal throughput

## Future Enhancements

- [ ] Pipeline orchestration for multi-provider workflows
- [ ] Quality metrics integration (BLEU, chrF, TER)
- [ ] Checkpoint/resume for large translations
- [ ] Directory structure preservation
- [ ] Format processors (CoNLL-U, JSON, XML)
- [ ] CLI interface
- [ ] Web UI

## License

See main project LICENSE file.

## Contributing

1. Add new providers in `providers/`
2. Create prompt templates in `config/prompts/`
3. Add examples in `data/examples/`
4. Update this README

## Support

For issues or questions, see main project repository.
