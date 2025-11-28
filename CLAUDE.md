# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BhaashikLowsResLangTrans is a cost-optimized translation system for low-resource Indic languages. It creates synthetic parallel corpora using a tiered strategy that combines free open-source models (IndicTrans2, NLLB) with selective commercial API enhancement (Claude).

**Key Languages:**
- **22 Supported (FREE)**: Direct IndicTrans2 support for scheduled Indian languages
- **6 Low-Resource (Hindi Pivot)**: Bhojpuri, Magahi, Awadhi, Braj, Marwari, Bundeli

**Cost Strategy**: 70% free (Hindi pivot) + 20% Claude Haiku 3.5 + 10% Claude Haiku 4.5

## Environment Setup

### Quick Start

```bash
# Create conda environment
conda env create -f environment.yml
conda activate NLPLResourceDownload

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with BASE_DIR and API keys (if using commercial APIs)
```

### Required Environment Variables

```bash
export BASE_DIR="/mnt/data/nlp_resources"  # Main data directory
export HF_DATASETS_CACHE="$BASE_DIR/cache/datasets"
export HF_HOME="$BASE_DIR/cache/huggingface"
export TRANSFORMERS_CACHE="$BASE_DIR/cache/transformers"
export HF_HUB_ENABLE_HF_TRANSFER=1
export ANTHROPIC_API_KEY="..."  # Only if using Claude API
```

## Common Commands

### Translation

```bash
# Translate using IndicTrans2 (free, for 22 supported languages)
python scripts/translate.py --src en --tgt hi --text "Hello world" --indictrans2

# Translate using Hindi pivot (free, for unsupported languages)
python scripts/translate.py --src en --tgt bho --input texts.txt --output out.txt --hindi-pivot

# Translate using tiered strategy (costs money - 70/20/10)
python scripts/translate.py --src en --tgt bho --input texts.txt --output out.txt --tiered

# Translate entire corpus with directory structure preservation
python scripts/translate_corpus.py \
  --input-dir /path/to/UD_Hindi \
  --output-dir /path/to/UD_Bhojpuri \
  --src hi --tgt bho --format conllu --hindi-pivot
```

### Cost Estimation

```bash
# Estimate cost before running tiered translation
python scripts/translate.py --src en --tgt bho --num-samples 10000 --estimate-only

# View cost tracking log
python -c "from src.utils.cost_tracker import CostTracker; \
           CostTracker('logs/cost_tracking.json').print_summary()"
```

### Downloading Resources

```bash
# Download specific models or datasets
python scripts/download.py --models indictrans2
python scripts/download.py --datasets samanantar --languages hi bn

# Verify setup
python scripts/verify.py --all
python scripts/verify.py --environment
python scripts/verify.py --models
```

### Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_config.py -v

# Run examples
python examples/example_1_basic_translation.py
python examples/example_2_hindi_pivot.py
python examples/example_3_tiered_strategy.py
python examples/example_5_corpus_translation.py
```

## Architecture

### Core Translation Flow

```
User Request
    ↓
scripts/translate.py (CLI interface)
    ↓
TieredOrchestrator (decides routing based on tier percentages)
    ↓
├─→ HindiPivotTranslator (70%) - uses IndicTrans2Translator
├─→ ClaudeTranslator (20%, Haiku 3.5)
└─→ ClaudeTranslator (10%, Haiku 4.5)
    ↓
CostTracker (logs API usage)
```

### Key Components

**1. Translation Strategies (src/translators/)**

- `IndicTrans2Translator`: Primary free translator for 22 scheduled languages
  - Handles en↔indic and indic↔indic translation
  - Model loading with lazy initialization
  - Batch processing with GPU support

- `HindiPivotTranslator`: Free translation for unsupported languages
  - Uses IndicTrans2 with Hindi as pivot: `en → hi → target_lang` (approximation)
  - NOTE: Hindi output is used as approximation for related Indo-Aryan languages
  - Linguistically valid for Bhojpuri, Magahi, Awadhi, Braj, Marwari, Bundeli

- `NLLBTranslator`: Alternative free model with broader language coverage
  - Supports 200+ languages including some low-resource Indic languages

**2. Tiered Orchestrator (src/tiered_orchestrator.py)**

The `TieredOrchestrator` class implements the cost-optimization strategy:
- Randomly assigns tier labels to input texts based on percentages (default 70/20/10)
- Routes texts to appropriate translator
- Aggregates results and tracks costs
- Key method: `translate(texts, src_lang, tgt_lang)` returns translations with metadata

**3. API Clients (src/api_clients/)**

- `ClaudeTranslator`: Wrapper for Anthropic Claude API
  - Supports multiple models (haiku_3, haiku_3_5, haiku_4_5, sonnet_4_5)
  - Automatic cost tracking via `CostTracker`
  - Batch processing with rate limiting

**4. Utilities (src/utils/)**

- `Config`: Centralized configuration from `config/config.yaml` and environment variables
  - Language definitions (supported vs unsupported)
  - Model paths and settings
  - Tier percentages and costs
  - Access via: `config.get('path.to.setting', default_value)`

- `CostTracker`: Real-time API cost monitoring
  - Logs to JSON file (default: `logs/cost_tracking.json`)
  - Tracks input/output tokens, costs per model, timestamps
  - Budget alerts if enabled in config
  - Key methods: `track_usage()`, `get_statistics()`, `print_summary()`

- `logger`: Structured logging with loguru
  - Console output with rich formatting
  - File logging to `logs/translation.log`
  - Use `get_logger(__name__)` in modules

**5. Quality Assessment (src/quality/)**

- `QualityMetrics`: BLEU, chrF, TER scoring using sacrebleu
  - `calculate_all_metrics(hypotheses, references)` returns scores
  - Useful for evaluating translation quality

**6. Data Downloaders (src/downloaders/)**

- `SamanantarDownloader`: Downloads parallel corpus from HuggingFace (49.7M pairs)
- `ModelDownloader`: Downloads IndicTrans2, NLLB, and other models from HF

### Configuration System

Configuration is hierarchical:
1. `config/config.yaml` - base configuration
2. `.env` - environment-specific settings (BASE_DIR, API keys)
3. Command-line arguments override both

**Key config sections:**
- `languages.indictrans2_supported`: List of 22 free languages
- `languages.unsupported`: Dict of 6 languages needing Hindi pivot
- `strategy.unsupported_strategy.tiers`: Tier percentages and costs
- `api.anthropic.models`: Pricing per model (INR per 1M tokens)
- `processing`: Batch size, max_length, num_beams, GPU settings

### Hindi Pivot Strategy

For unsupported Indo-Aryan languages (Bhojpuri, Magahi, Awadhi, Braj, Marwari, Bundeli):

**Why it works:**
- All 6 target languages belong to the Indo-Aryan family
- High lexical overlap with Hindi (50-80%)
- Similar grammar, syntax, and morphology
- Hindi acts as a close approximation

**Translation path:**
- `en → hi` (IndicTrans2, high quality)
- `hi → target_lang` (currently returns Hindi as approximation)
- For actual target language: use Claude API for final step (via tiered strategy)

**Implementation detail:**
The `HindiPivotTranslator` currently returns Hindi as the output because these languages are linguistically close. For production quality, the tiered strategy applies Claude API enhancement to convert Hindi → actual target language.

### Corpus Translation

The `scripts/translate_corpus.py` supports structured dataset translation:

**Features:**
- Preserves directory structure from input to output
- Supports plain text (.txt) and CoNLL-U (.conllu) formats
- CoNLL-U: Translates `# text = ...` metadata lines, preserves annotation structure
- Progress tracking with checkpointing
- Resumable translation (skips already translated files)

**Use case:** Universal Dependencies (UD) corpus adaptation
- Input: `UD_Hindi/`
- Output: `UD_Bhojpuri/` with same file structure, translated text

## Project File Structure

```
src/
├── translators/          # Translation implementations
│   ├── indictrans2_translator.py
│   ├── nllb_translator.py
│   └── hindi_pivot_translator.py
├── api_clients/          # Commercial API wrappers
│   └── claude_client.py
├── downloaders/          # Dataset and model downloaders
├── quality/              # Quality metrics (BLEU, chrF, TER)
├── utils/                # Config, logging, cost tracking
└── tiered_orchestrator.py  # Main orchestration logic

scripts/
├── download.py           # Download datasets/models
├── translate.py          # Main CLI translation interface
├── translate_corpus.py   # Corpus-level translation
└── verify.py            # Setup verification

config/
└── config.yaml          # Central configuration

examples/                 # Python usage examples
tests/                    # pytest test suite
```

## Development Notes

### Adding New Translation Method

1. Create translator class in `src/translators/`
2. Implement `translate(texts, src_lang, tgt_lang)` method
3. Add to `TieredOrchestrator` if using in tiered strategy
4. Update `config/config.yaml` with pricing (if applicable)

### Adding New Language

**For IndicTrans2-supported language:**
- Add to `config/config.yaml` under `languages.indictrans2_supported`

**For unsupported language:**
- Add to `config/config.yaml` under `languages.unsupported`
- Specify pivot language (usually 'hi' for Indo-Aryan)
- Set speaker count and language family

### Cost Tracking Integration

When adding new API clients, integrate `CostTracker`:

```python
from src.utils.cost_tracker import CostTracker

tracker = CostTracker(log_file='logs/cost_tracking.json')
tracker.track_usage(
    model='haiku_3_5',
    input_tokens=1000,
    output_tokens=500,
    metadata={'src_lang': 'en', 'tgt_lang': 'bho'}
)
```

### Working with Config

```python
from src.utils.config import Config

config = Config()  # Loads from config/config.yaml + .env

# Get values with default
batch_size = config.get('processing.batch_size', 32)

# Get language lists
supported = config.get_indictrans2_supported_languages()
unsupported = config.get_unsupported_languages()
```

## Data Sources

**Samanantar Corpus**: 49.7M parallel sentence pairs
- Dataset: `ai4bharat/samanantar`
- Languages: as, bn, gu, hi, kn, ml, mr, or, pa, ta, te
- Size: ~60 GB uncompressed

**Models:**
- IndicTrans2: `ai4bharat/indictrans2-*` (1B and 200M variants)
- NLLB: `facebook/nllb-200-*` (600M, 1.3B, 3.3B)

## References

- **IndicTrans2 Paper**: https://arxiv.org/abs/2305.16307
- **Samanantar Paper**: https://arxiv.org/abs/2104.05596
- **AI4Bharat**: https://ai4bharat.org/
- **Claude API Docs**: https://docs.anthropic.com/claude/docs
