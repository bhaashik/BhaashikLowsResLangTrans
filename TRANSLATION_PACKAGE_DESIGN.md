# Universal Translation Package Design

## Overview

A flexible, extensible translation framework that supports:
- Multiple translation providers (LLMs, Translation APIs, Local Models)
- Any language pair
- Custom prompting strategies with external configuration
- Directory structure preservation
- Format-agnostic processing (text, CoNLL-U, JSON, etc.)
- Cost tracking and quality metrics

## Architecture

```
universal_translate/
├── core/
│   ├── __init__.py
│   ├── base_translator.py      # Abstract base class for all translators
│   ├── translation_request.py  # Request/Response models
│   ├── translation_pipeline.py # Orchestration layer
│   └── provider_registry.py    # Dynamic provider registration
│
├── providers/
│   ├── __init__.py
│   ├── base_provider.py        # Abstract provider interface
│   ├── anthropic_provider.py   # Claude API
│   ├── openai_provider.py      # OpenAI/ChatGPT
│   ├── google_provider.py      # Google Cloud Translate
│   ├── azure_provider.py       # Azure Translator
│   ├── deepl_provider.py       # DeepL API
│   ├── indictrans_provider.py  # Local IndicTrans2
│   └── nllb_provider.py        # Local NLLB
│
├── prompts/
│   ├── __init__.py
│   ├── prompt_manager.py       # Load/manage prompts from configs
│   ├── prompt_template.py      # Template rendering engine
│   └── cache_manager.py        # Prompt caching support
│
├── processors/
│   ├── __init__.py
│   ├── base_processor.py       # Abstract file processor
│   ├── text_processor.py       # Plain text files
│   ├── conllu_processor.py     # CoNLL-U format
│   ├── json_processor.py       # JSON data
│   └── xml_processor.py        # XML/TMX formats
│
├── utils/
│   ├── __init__.py
│   ├── file_handler.py         # Directory traversal, structure preservation
│   ├── cost_tracker.py         # Track API costs
│   ├── quality_metrics.py      # BLEU, chrF, etc.
│   ├── rate_limiter.py         # API rate limiting
│   └── checkpoint.py           # Resume capability
│
├── config/
│   ├── providers.yaml          # Provider configurations
│   ├── prompts/                # Prompt templates by use case
│   │   ├── general.yaml
│   │   ├── low_resource.yaml
│   │   ├── domain_specific.yaml
│   │   └── with_examples.yaml
│   └── language_pairs.yaml     # Language-specific settings
│
└── cli/
    ├── __init__.py
    └── translate.py            # Command-line interface
```

## Key Design Principles

### 1. Provider Abstraction

All translation providers implement a common interface:

```python
class BaseProvider(ABC):
    @abstractmethod
    async def translate(self, request: TranslationRequest) -> TranslationResponse:
        """Translate text using this provider."""
        pass

    @abstractmethod
    def get_cost_estimate(self, request: TranslationRequest) -> CostEstimate:
        """Estimate cost before translation."""
        pass

    @abstractmethod
    def supports_batch(self) -> bool:
        """Whether provider supports batch translation."""
        pass
```

### 2. External Configuration

**No hardcoded prompts or examples**. Everything loaded from YAML/JSON:

```yaml
# config/prompts/low_resource.yaml
name: "low_resource_translation"
description: "Translation for low-resource Indic languages"

system_prompt: |
  You are an expert translator specializing in {source_lang} to {target_lang} translation.
  Focus on preserving meaning, cultural context, and linguistic nuances.

user_prompt: |
  Translate the following {source_lang} text to {target_lang}:

  {text}

examples_file: "data/examples/hi_to_bho_examples.json"  # Optional
use_prompt_caching: true  # For providers that support it

parameters:
  temperature: 0.3
  max_tokens: 2048
```

### 3. Flexible Pipeline

```python
pipeline = TranslationPipeline(
    provider="anthropic",  # or "google", "openai", etc.
    model="claude-haiku-4.5",
    prompt_config="config/prompts/low_resource.yaml",
    processor="text",  # or "conllu", "json"
    cost_tracker=True,
    checkpoint_dir="checkpoints/"
)

results = pipeline.translate_directory(
    input_dir="input/Hindi/",
    output_dir="output/Bhojpuri/",
    src_lang="hi",
    tgt_lang="bho",
    preserve_structure=True
)
```

### 4. Processor Pattern

Different file formats handled by specialized processors:

```python
class TextProcessor(BaseProcessor):
    """Process plain text files line by line."""

    def read(self, filepath: Path) -> List[str]:
        """Read and split into translatable units."""
        pass

    def write(self, filepath: Path, translations: List[str]):
        """Write translations maintaining format."""
        pass
```

### 5. Dynamic Examples Loading

```json
// data/examples/hi_to_bho_examples.json
{
  "examples": [
    {
      "source": "राम घर जा रहे हैं।",
      "target": "राम घर जात बाड़े।",
      "context": "present continuous"
    },
    {
      "source": "यह किताब अच्छी है।",
      "target": "ई किताब बढ़िया बा।",
      "context": "simple statement"
    }
  ],
  "metadata": {
    "source_lang": "hi",
    "target_lang": "bho",
    "domain": "general",
    "created_by": "human_annotator",
    "created_date": "2025-12-01"
  }
}
```

## Configuration Files

### providers.yaml

```yaml
providers:
  anthropic:
    api_key_env: "ANTHROPIC_API_KEY"
    models:
      claude-haiku-3.5:
        input_cost: 0.68  # per 1M tokens (INR)
        output_cost: 3.40
        supports_caching: true
        cache_discount: 0.9
      claude-haiku-4.5:
        input_cost: 0.85
        output_cost: 4.25
        supports_caching: true
    batch_api:
      enabled: true
      discount: 0.5
    rate_limits:
      requests_per_minute: 1000
      tokens_per_minute: 100000

  openai:
    api_key_env: "OPENAI_API_KEY"
    models:
      gpt-4o:
        input_cost: 2.50  # USD
        output_cost: 10.00
      gpt-4o-mini:
        input_cost: 0.15
        output_cost: 0.60
    rate_limits:
      requests_per_minute: 500

  google:
    credentials_env: "GOOGLE_APPLICATION_CREDENTIALS"
    api: "translate_v3"
    cost_per_char: 20  # USD per 1M chars
    supports_batch: true

  local:
    indictrans2:
      model_path: "ai4bharat/indictrans2-en-indic-dist-200M"
      device: "cuda"
      batch_size: 32
      cost_per_char: 0  # Free

    nllb:
      model_path: "facebook/nllb-200-distilled-600M"
      device: "cuda"
      batch_size: 16
      cost_per_char: 0
```

### language_pairs.yaml

```yaml
language_pairs:
  hi_to_bho:
    source: "hi"
    target: "bho"
    target_name: "Bhojpuri"
    strategy: "low_resource_pivot"
    recommended_prompts:
      - "prompts/low_resource.yaml"
      - "prompts/indo_aryan.yaml"
    example_corpus: "data/examples/hi_to_bho_examples.json"
    quality_threshold: 0.7

  en_to_hi:
    source: "en"
    target: "hi"
    target_name: "Hindi"
    strategy: "direct"
    recommended_providers:
      - "indictrans2"  # Free, high quality
      - "google"
    quality_threshold: 0.8
```

## Usage Examples

### CLI Usage

```bash
# Translate directory using Claude with custom prompt
universal-translate \
  --provider anthropic \
  --model claude-haiku-4.5 \
  --prompt-config config/prompts/low_resource.yaml \
  --input input/Hindi/plain-text/by_domain/ \
  --output output/Bhojpuri/plain-text/by_domain/ \
  --src-lang hi \
  --tgt-lang bho \
  --preserve-structure \
  --track-costs \
  --checkpoint-dir checkpoints/

# Use Google Translate (no prompting needed)
universal-translate \
  --provider google \
  --input input/English/ \
  --output output/Hindi/ \
  --src-lang en \
  --tgt-lang hi

# Use local IndicTrans2 (free)
universal-translate \
  --provider local:indictrans2 \
  --input input/English/ \
  --output output/Bengali/ \
  --src-lang en \
  --tgt-lang bn
```

### Python API

```python
from universal_translate import TranslationPipeline, PromptManager
from universal_translate.providers import AnthropicProvider

# Load prompt configuration
prompt_mgr = PromptManager("config/prompts/low_resource.yaml")
prompt_mgr.load_examples("data/examples/hi_to_bho_examples.json")

# Create provider
provider = AnthropicProvider(
    model="claude-haiku-4.5",
    prompt_manager=prompt_mgr
)

# Create pipeline
pipeline = TranslationPipeline(
    provider=provider,
    processor="text",
    src_lang="hi",
    tgt_lang="bho"
)

# Translate preserving structure
results = pipeline.translate_directory(
    input_dir="input/Hindi/by_domain/",
    output_dir="output/Bhojpuri/by_domain/",
    preserve_structure=True,
    resume=True  # Resume from checkpoint if exists
)

# Access results
print(f"Translated: {results.total_sentences}")
print(f"Cost: ₹{results.total_cost:.2f}")
print(f"Quality Score: {results.avg_quality:.2f}")
```

## Features

### 1. Prompt Caching Support

For providers supporting prompt caching (Claude):
- System prompts cached automatically
- Examples cached across requests
- 90% cost reduction on cached content

### 2. Cost Tracking

Automatic tracking across all providers:
- Per-request costs
- Aggregate by provider, model, language pair
- Budget alerts
- Cost estimation before translation

### 3. Quality Metrics

Built-in quality assessment:
- BLEU scores (if references available)
- chrF scores
- Custom metrics per language pair

### 4. Checkpointing

Resume interrupted translations:
- Save progress every N requests
- Skip already-translated files
- Atomic file writes

### 5. Rate Limiting

Automatic rate limiting per provider:
- Request-based limits
- Token-based limits
- Exponential backoff on errors

### 6. Batch Processing

Efficient batch handling:
- Automatic batching for supported providers
- Parallel processing for local models
- Memory-efficient streaming

## Extensibility

### Adding New Provider

```python
# providers/custom_provider.py
from universal_translate.providers import BaseProvider
from universal_translate.core import TranslationRequest, TranslationResponse

class CustomProvider(BaseProvider):
    def __init__(self, api_key: str, **kwargs):
        super().__init__(**kwargs)
        self.client = CustomAPIClient(api_key)

    async def translate(self, request: TranslationRequest) -> TranslationResponse:
        # Implement translation logic
        result = await self.client.translate(
            text=request.text,
            source=request.src_lang,
            target=request.tgt_lang
        )

        return TranslationResponse(
            text=result.translated_text,
            cost=self.calculate_cost(request, result),
            metadata={"provider": "custom"}
        )
```

### Adding New Processor

```python
# processors/custom_processor.py
from universal_translate.processors import BaseProcessor

class XMLProcessor(BaseProcessor):
    def read(self, filepath: Path) -> List[TranslationUnit]:
        # Parse XML, extract translatable elements
        pass

    def write(self, filepath: Path, units: List[TranslationUnit]):
        # Write back to XML maintaining structure
        pass
```

## Migration from Current Codebase

1. **Extract generic components** from existing code:
   - `src/utils/cost_tracker.py` → `universal_translate/utils/cost_tracker.py`
   - `src/quality/metrics.py` → `universal_translate/utils/quality_metrics.py`

2. **Refactor providers**:
   - `src/api_clients/claude_client.py` → `universal_translate/providers/anthropic_provider.py`
   - `src/translators/indictrans2_translator.py` → `universal_translate/providers/indictrans_provider.py`

3. **Extract prompts** from code to YAML configs

4. **Create abstraction layers** for current specific implementations

## Benefits

1. **Flexibility**: Easy to switch between providers
2. **Maintainability**: Clear separation of concerns
3. **Extensibility**: Add new providers/processors without modifying core
4. **Testability**: Mock providers for testing
5. **Reusability**: Use for any translation project
6. **Cost-Efficiency**: Compare costs across providers
7. **Quality-Focused**: Built-in metrics and validation
