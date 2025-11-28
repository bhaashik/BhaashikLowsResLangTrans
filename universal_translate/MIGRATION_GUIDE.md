# Migration Guide: From Current Codebase to Universal Translate

This guide explains how to migrate from the current project-specific translation code to the universal translation package.

## Key Differences

### Before (Current Code)

```python
# Hardcoded prompts in Python files
from src.api_clients import ClaudeTranslator

translator = ClaudeTranslator(model='haiku_4_5')
translations = translator.translate(
    texts,
    src_lang='hi',
    tgt_lang='bho',
    # Prompt embedded in code
)
```

### After (Universal Translate)

```python
# Prompts in external YAML files
from universal_translate import AnthropicProvider, PromptManager

prompt_mgr = PromptManager("config/prompts/low_resource.yaml")
provider = AnthropicProvider(model="claude-haiku-4.5", prompt_manager=prompt_mgr)

request = TranslationRequest(units=units, src_lang='hi', tgt_lang='bho')
response = provider.translate_sync(request)
```

## Step-by-Step Migration

### 1. Extract Prompts to YAML

**Current code** (in `src/api_clients/claude_client.py`):
```python
system_prompt = f"""You are an expert translator...
Translate {src_lang} to {tgt_lang}..."""

user_prompt = f"Translate: {text}"
```

**New YAML** (`config/prompts/my_prompt.yaml`):
```yaml
system_prompt: |
  You are an expert translator...
  Translate ${source_lang} to ${target_lang}...

user_prompt: |
  Translate: ${text}
```

### 2. Extract Examples to JSON

**Current code**:
```python
examples = [
    {"source": "राम घर जा रहे हैं।", "target": "राम घर जात बाड़े।"},
    # More examples hardcoded...
]
```

**New JSON** (`data/examples/hi_to_bho.json`):
```json
{
  "examples": [
    {
      "source": "राम घर जा रहे हैं।",
      "target": "राम घर जात बाड़े।",
      "source_lang": "hi",
      "target_lang": "bho"
    }
  ]
}
```

### 3. Migrate Provider Code

**Current** (`src/api_clients/claude_client.py`):
```python
class ClaudeTranslator:
    def __init__(self, model, config, cost_tracker):
        self.model = model
        self.config = config
        self.cost_tracker = cost_tracker
        self.client = anthropic.Anthropic()

    def translate(self, texts, src_lang, tgt_lang):
        # Provider-specific implementation
        pass
```

**New** (using `AnthropicProvider`):
```python
from universal_translate import AnthropicProvider

provider = AnthropicProvider(model="claude-haiku-4.5")
# All provider-specific logic handled internally
```

### 4. Update Translation Calls

**Current**:
```python
from src.translators import HindiPivotTranslator

translator = HindiPivotTranslator()
translations = translator.translate(
    texts,
    src_lang='en',
    tgt_lang='bho',
    batch_size=32
)
```

**New**:
```python
from universal_translate import AnthropicProvider, TranslationRequest, TranslationUnit

# Create units
units = [TranslationUnit(text=t, index=i) for i, t in enumerate(texts)]

# Create request
request = TranslationRequest(units=units, src_lang='en', tgt_lang='bho')

# Translate
provider = AnthropicProvider(model="claude-haiku-4.5")
with provider:
    response = provider.translate_sync(request)

# Extract translations
translations = [r.translation for r in response.results]
```

## Component Mapping

| Current Component | Universal Translate | Notes |
|------------------|---------------------|-------|
| `src/api_clients/claude_client.py` | `providers/anthropic_provider.py` | Refactored with unified interface |
| `src/translators/indictrans2_translator.py` | `providers/indictrans_provider.py` | (To be implemented) |
| `src/utils/cost_tracker.py` | Built into providers | Automatic cost tracking |
| `src/quality/metrics.py` | `utils/quality_metrics.py` | (To be implemented) |
| Hardcoded prompts | `config/prompts/*.yaml` | External configuration |
| Hardcoded examples | `data/examples/*.json` | External data files |

## Benefits of Migration

### 1. Flexibility

**Before**: Changing prompts requires code modification
```python
# Edit Python file
def translate(...):
    prompt = "Translate..."  # Modify this
```

**After**: Just edit YAML file
```yaml
# Edit config/prompts/my_prompt.yaml
user_prompt: "New prompt template..."
```

### 2. Reusability

**Before**: Translation code tied to specific project structure

**After**: Universal package usable in any project
```python
# Use in any Python project
from universal_translate import AnthropicProvider
```

### 3. Testability

**Before**: Hard to test without actual API calls

**After**: Easy to mock providers
```python
# Mock provider for testing
class MockProvider(BaseTranslator):
    def translate(self, request):
        # Return fake translations for testing
        pass
```

### 4. Cost Optimization

**Before**: Manual prompt caching setup

**After**: Automatic caching support
```yaml
# Just set this in config
use_prompt_caching: true
```

### 5. Multi-Provider Support

**Before**: Separate implementations for each provider

**After**: Unified interface for all providers
```python
# Switch providers easily
provider = AnthropicProvider(...)  # or
provider = GoogleProvider(...)     # or
provider = IndicTrans2Provider(...) # Same interface!
```

## Compatibility Layer

To maintain backward compatibility during migration, create adapter functions:

```python
# compatibility.py - Use old API with new backend

from universal_translate import AnthropicProvider, TranslationRequest, TranslationUnit

def translate_legacy(texts, src_lang, tgt_lang, model='haiku_4_5'):
    """Legacy function signature using new backend."""
    # Map to new API
    units = [TranslationUnit(text=t, index=i) for i, t in enumerate(texts)]
    request = TranslationRequest(units=units, src_lang=src_lang, tgt_lang=tgt_lang)

    # Use new provider
    provider = AnthropicProvider(model=model)
    with provider:
        response = provider.translate_sync(request)

    # Return in old format
    return [r.translation for r in response.results]
```

## Gradual Migration Strategy

1. **Phase 1**: Extract prompts to YAML files
   - Move all prompt strings to `config/prompts/`
   - Update code to load from YAML

2. **Phase 2**: Extract examples to JSON
   - Move example data to `data/examples/`
   - Update code to load from JSON

3. **Phase 3**: Adopt provider interfaces
   - Replace direct API calls with provider classes
   - Keep existing function signatures

4. **Phase 4**: Migrate to TranslationRequest/Response
   - Update calling code to use new data models
   - Remove legacy adapters

5. **Phase 5**: Full migration
   - Use universal_translate package exclusively
   - Remove old translation code

## Testing Migration

```python
# test_migration.py
from universal_translate import AnthropicProvider, TranslationRequest, TranslationUnit

def test_backward_compatibility():
    """Test that new code produces same results as old code."""
    # Old way (if still available)
    # old_result = old_translator.translate("test", "en", "hi")

    # New way
    request = TranslationRequest(
        units=[TranslationUnit(text="test", index=0)],
        src_lang="en",
        tgt_lang="hi"
    )
    provider = AnthropicProvider(model="claude-haiku-4.5")
    with provider:
        response = provider.translate_sync(request)
    new_result = response.results[0].translation

    # Compare (if testing with mocked providers)
    # assert old_result == new_result
```

## Common Migration Issues

### Issue 1: Different parameter names

**Problem**: Old code uses `src_lang`, new uses `source_lang`

**Solution**: Check prompt templates for correct variable names
```yaml
# Use ${source_lang} or ${src_lang} consistently
user_prompt: "Translate from ${src_lang}..."
```

### Issue 2: Batch processing differences

**Problem**: Old code batches internally, new uses TranslationRequest

**Solution**: Create units from batch
```python
# Old: translator.translate(batch, ...)
# New:
units = [TranslationUnit(text=t, index=i) for i, t in enumerate(batch)]
request = TranslationRequest(units=units, ...)
```

### Issue 3: Cost tracking format

**Problem**: Old CostTracker has different schema

**Solution**: Adapt to new format
```python
# Old: cost_tracker.track_usage(...)
# New: Cost automatically tracked in response.total_cost
for result in response.results:
    print(f"Cost: {result.cost}")
```

## Next Steps

1. Review current codebase to identify all hardcoded prompts
2. Create YAML configs for each use case
3. Extract translation examples to JSON
4. Start with one module (e.g., Claude API) and migrate
5. Test thoroughly before migrating other modules
6. Update documentation to reflect new usage

## Support

For questions or issues during migration, refer to:
- `universal_translate/README.md` - Package documentation
- `universal_translate/examples/` - Usage examples
- `TRANSLATION_PACKAGE_DESIGN.md` - Architecture details
