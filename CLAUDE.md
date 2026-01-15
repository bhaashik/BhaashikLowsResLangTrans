# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BhaashikLowsResLangTrans is a dual-architecture translation system for low-resource Indic languages:

1. **Legacy System** (`src/`): Original tiered strategy using IndicTrans2 + Claude API
2. **Universal Translation Framework** (`universal_translate/`): Modern plugin-based architecture supporting multiple LLM providers (OpenAI, Anthropic, Gemini, Vertex AI)

### Target Languages

**22 Free Languages** (via IndicTrans2): Assamese, Bengali, Gujarati, Hindi, Kannada, Kashmiri, Konkani, Malayalam, Manipuri, Marathi, Nepali, Odia, Punjabi, Sanskrit, Sindhi, Tamil, Telugu, Urdu, Bodo, Santhali, Maithili, Dogri

**6 Low-Resource Languages** (requiring API): Bhojpuri (bho), Magahi (mag), Awadhi (awa), Braj (bra), Marwari (mwr), Bundeli (bns)

## Architecture

### Two Translation Systems

**1. Legacy Tiered System (`src/`)**
- `src/tiered_orchestrator.py`: Implements 70/20/10 strategy (Hindi pivot / Claude 3.5 / Claude 4.5)
- `src/translators/`: IndicTrans2, NLLB, Hindi pivot implementations
- `src/api_clients/`: Claude API client with cost tracking
- `src/quality/`: BLEU, chrF, TER metrics
- `src/utils/`: Config, logger, cost tracker

**2. Universal Framework (`universal_translate/`)**
- `universal_translate/core/`: Abstract `BaseTranslator` interface and request/response models
- `universal_translate/providers/`: Provider implementations (OpenAI, Anthropic, Gemini, Vertex)
- `universal_translate/providers/registry.py`: Dynamic provider registration and discovery
- `universal_translate/prompts/`: Multi-language prompt templates with examples
- `universal_translate/processors/`: Input/output format handlers (CoNLL-U, plain text)

### Key Design Patterns

**Provider Plugin System**: All providers implement `BaseTranslator` interface defined in `universal_translate/core/base_translator.py`. The registry pattern in `universal_translate/providers/registry.py` allows runtime provider discovery and switching.

**Translation Request Flow**:
```
TranslationRequest → BaseTranslator.translate() → Provider-specific API call → TranslationResponse
```

**Prompt Caching Strategy**: Each provider implements automatic prompt caching:
- OpenAI: 50% discount on cached prompts (ephemeral storage)
- Anthropic: 90% discount on cached prompts (persistent)
- Gemini: 75% discount on cached prompts

**Cost Tracking**: Automatic cost calculation across all providers with provider-specific pricing models defined in each provider class.

## Common Commands

### Environment Setup

```bash
# Create conda environment
conda env create -f environment.yml
conda activate NLPLResourceDownload

# Or install via pip
pip install -r requirements.txt

# Setup API keys (REQUIRED)
cp .env.example .env
# Edit .env with your actual API keys
```

### Running Translations

**Universal CLI (Recommended for new work)**:
```bash
# List available providers
python scripts/translate_cli.py --list-providers

# OpenAI GPT-4o-mini
python scripts/translate_cli.py --provider openai --source-lang hi --target-lang mwr --text "नमस्ते"

# Anthropic Claude
python scripts/translate_cli.py --provider anthropic --model claude-haiku-4.5 --source-lang hi --target-lang bho --text "नमस्ते"

# Google Gemini
python scripts/translate_cli.py --provider gemini --source-lang hi --target-lang awa --text "नमस्ते"

# Batch translation from file
python scripts/translate_cli.py --provider openai --source-lang hi --target-lang mwr --input input.txt --output output.txt

# Estimate costs before running
python scripts/translate_cli.py --provider openai --source-lang hi --target-lang mwr --input input.txt --estimate-only
```

**Legacy CLI (for IndicTrans2 and tiered strategy)**:
```bash
# Free IndicTrans2 translation
python scripts/translate.py --src en --tgt hi --text "Hello" --indictrans2

# Hindi pivot for low-resource languages (FREE)
python scripts/translate.py --src en --tgt bho --text "Hello" --hindi-pivot

# Tiered strategy (70/20/10 - COSTS MONEY)
python scripts/translate.py --src en --tgt bho --input texts.txt --output translations.txt --tiered

# Cost estimation
python scripts/translate.py --src en --tgt bho --num-samples 10000 --estimate-only
```

**Corpus Translation** (preserves directory structure):
```bash
python scripts/translate_corpus.py \
  --input-dir /path/to/UD_Hindi \
  --output-dir /path/to/UD_Bhojpuri \
  --src hi --tgt bho \
  --format conllu \
  --hindi-pivot
```

### Testing

```bash
# Run tests
pytest tests/

# Test specific module
pytest tests/test_config.py

# Run with coverage
pytest --cov=src --cov=universal_translate tests/

# Test all providers (requires API keys)
python scripts/test_all_providers.py
```

### Verification and Monitoring

```bash
# Verify setup
python scripts/verify.py --all

# View cost tracking summary
python -c "from src.utils.cost_tracker import CostTracker; CostTracker('logs/cost_tracking.json').print_summary()"
```

### Code Quality

```bash
# Format code
black src/ universal_translate/ scripts/ examples/

# Lint
flake8 src/ universal_translate/ scripts/

# Type checking
mypy src/ universal_translate/
```

## Configuration

### Main Config File: `config/config.yaml`

- **languages**: Defines supported and unsupported languages with pivot strategies
- **models**: IndicTrans2, NLLB model paths
- **strategy**: Tiered approach configuration (percentages: free/enhancement/premium)
- **api**: Provider-specific pricing (Anthropic, OpenAI, Google, Azure)
- **processing**: Batch size, GPU settings, checkpointing
- **cost_tracking**: Budget alerts, currency conversion

### Environment Variables (`.env`)

Required for API access:
```bash
OPENAI_API_KEY=sk-proj-...
ANTHROPIC_API_KEY=sk-ant-api03-...
GOOGLE_API_KEY=...
GOOGLE_CLOUD_PROJECT=...  # For Vertex AI
GOOGLE_APPLICATION_CREDENTIALS=...  # For Vertex AI
```

Optional for local model paths:
```bash
BASE_DIR=/mnt/data/nlp_resources
HF_DATASETS_CACHE=$BASE_DIR/cache/datasets
HF_HOME=$BASE_DIR/cache/huggingface
TRANSFORMERS_CACHE=$BASE_DIR/cache/transformers
```

### Prompt Configuration

Multi-language prompt templates in `universal_translate/config/prompts/`:
- `awadhi-prompt.yaml`, `braj-prompt.yaml`, `marwari-prompt.yaml`, etc.
- Each file contains language-specific examples and cultural context
- Prompts are automatically loaded by `universal_translate/prompts/prompt_manager.py`

## Python API Usage

### Universal Framework (Recommended)

```python
from universal_translate.providers import OpenAIProvider, AnthropicProvider
from universal_translate.core import TranslationRequest

# Initialize provider
translator = OpenAIProvider(model='gpt-4o-mini')

# Create request
request = TranslationRequest(
    units=["नमस्ते", "आप कैसे हैं?"],
    source_lang="hi",
    target_lang="mwr"
)

# Translate
response = translator.translate_sync(request)
print(response.translations)
print(f"Cost: {response.cost_info.total_cost} {response.cost_info.currency}")
```

### Legacy API

```python
from src.translators import IndicTrans2Translator, HindiPivotTranslator
from src.tiered_orchestrator import TieredOrchestrator

# IndicTrans2 (free)
translator = IndicTrans2Translator()
result = translator.translate("Hello", src_lang='en', tgt_lang='hi')

# Hindi pivot (free)
pivot_translator = HindiPivotTranslator()
result = pivot_translator.translate(["Hello"], src_lang='en', tgt_lang='bho')

# Tiered orchestrator (costs money)
orchestrator = TieredOrchestrator()
results = orchestrator.translate(
    texts=["Hello", "Goodbye"],
    src_lang='en',
    tgt_lang='bho'
)
```

## Project Structure

```
.
├── src/                          # Legacy tiered translation system
│   ├── tiered_orchestrator.py   # 70/20/10 strategy orchestrator
│   ├── translators/              # IndicTrans2, NLLB, Hindi pivot
│   ├── api_clients/              # Claude API client
│   ├── quality/                  # Quality metrics (BLEU, chrF, TER)
│   └── utils/                    # Config, logger, cost tracker
├── universal_translate/          # Modern multi-provider framework
│   ├── core/                     # Abstract interfaces (BaseTranslator)
│   ├── providers/                # OpenAI, Anthropic, Gemini, Vertex
│   ├── prompts/                  # Language-specific prompt templates
│   ├── processors/               # Format handlers (CoNLL-U, text)
│   └── config/prompts/           # YAML prompt configurations
├── scripts/
│   ├── translate_cli.py          # Universal CLI (RECOMMENDED)
│   ├── translate.py              # Legacy CLI with tiered strategy
│   ├── translate_corpus.py       # Directory structure preservation
│   ├── download.py               # Download datasets/models
│   └── verify.py                 # Verify installation
├── examples/                     # Python usage examples
├── config/config.yaml            # Main configuration
├── environment.yml               # Conda environment
├── requirements.txt              # Pip dependencies
└── tests/                        # Test suite
```

## Cost Optimization

### Tiered Strategy (Legacy System)

- **Tier 1 (70%)**: Free Hindi pivot using IndicTrans2
- **Tier 2 (20%)**: Claude Haiku 3.5 (₹408/1M tokens)
- **Tier 3 (10%)**: Claude Haiku 4.5 (₹510/1M tokens)
- **Total for 12M pairs**: ~₹62,912 ($740)

### Provider Cost Comparison (Universal Framework)

- **OpenAI GPT-4o-mini**: $0.15/$0.60 per 1M tokens (input/output) - Best balance
- **Anthropic Claude Haiku 4.5**: $1/$5 per 1M tokens - Best quality, 90% cache discount
- **Gemini 2.0 Flash Exp**: Free tier available - Best for experimentation
- **Gemini 1.5 Flash**: $0.075/$0.30 per 1M tokens - Most economical paid option

### Caching Benefits

Prompts with examples are automatically cached:
- **OpenAI**: 50% discount on cached prompts (ephemeral storage, 5-10 min)
- **Anthropic**: 90% discount on cached prompts (5 minutes persistent)
- **Gemini**: 75% discount on cached prompts (1 hour persistent)

For batch translation, the prompt (with examples) is reused, drastically reducing costs.

## Important Notes for Development

### Adding a New Provider

1. Create `universal_translate/providers/{provider_name}_provider.py`
2. Inherit from `BaseTranslator` in `universal_translate/core/base_translator.py`
3. Implement required methods: `translate()`, `translate_sync()`, `get_cost_estimate()`
4. Add provider to registry in `universal_translate/providers/registry.py`
5. Update `scripts/translate_cli.py` to support new provider

### Adding a New Language

1. Create prompt template: `universal_translate/config/prompts/{language}-prompt.yaml`
2. Include 5-15 translation examples in the prompt
3. Add language metadata and cultural context
4. Test with `scripts/translate_cli.py --target-lang {language}`

### Modifying Tiered Strategy

Edit `config/config.yaml`:
```yaml
strategy:
  unsupported_strategy:
    tiers:
      - name: "free_pivot"
        percentage: 85  # Adjust percentages
      - name: "quality_enhancement"
        percentage: 10
      - name: "premium_quality"
        percentage: 5
```

### Translation Quality Assessment

Use `src/quality/metrics.py` for evaluation:
```python
from src.quality import QualityMetrics

metrics = QualityMetrics()
scores = metrics.calculate_all_metrics(
    hypotheses=translations,
    references=reference_translations
)
# Returns: corpus_bleu, chrf, ter scores
```

### Working with CoNLL-U Format

Use `scripts/translate_corpus.py` with `--format conllu` to preserve:
- Sentence boundaries
- Token indices
- Morphological features
- Dependency relations

Only the FORM (word) and LEMMA fields are translated; other annotations are preserved.

## Data Sources

- **Samanantar**: 49.7M English-Indic parallel pairs (`ai4bharat/samanantar`)
- **IndicTrans2 Models**: `ai4bharat/indictrans2-*` (1B and 200M variants)
- **NLLB Models**: `facebook/nllb-200-*` (600M, 1.3B, 3.3B variants)

Download via: `python scripts/download.py`

## References

- **IndicTrans2**: https://arxiv.org/abs/2305.16307
- **Samanantar**: https://arxiv.org/abs/2104.05596
- **NLLB**: https://arxiv.org/abs/2207.04672
- **OpenAI Prompt Caching**: https://platform.openai.com/docs/guides/prompt-caching
- **Anthropic Prompt Caching**: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
- **Gemini Context Caching**: https://ai.google.dev/gemini-api/docs/caching
