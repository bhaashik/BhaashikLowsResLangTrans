# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BhaashikLowsResLangTrans is a comprehensive translation system for low-resource Indic languages with three integrated components:

1. **Legacy System** (`src/`): Original tiered strategy using IndicTrans2 + Claude API
2. **Universal Translation Framework** (`universal_translate/`): Modern plugin-based architecture supporting multiple LLM providers (OpenAI, Anthropic, Gemini, Vertex AI, Fine-tuned Models)
3. **MT Training System** (`src/training/`): Complete fine-tuning infrastructure with LoRA, model merging, ensemble methods, and hybrid translation pipeline

### Target Languages

**22 Free Languages** (via IndicTrans2): Assamese, Bengali, Gujarati, Hindi, Kannada, Kashmiri, Konkani, Malayalam, Manipuri, Marathi, Nepali, Odia, Punjabi, Sanskrit, Sindhi, Tamil, Telugu, Urdu, Bodo, Santhali, Maithili, Dogri

**6 Low-Resource Languages** (requiring API): Bhojpuri (bho), Magahi (mag), Awadhi (awa), Braj (bra), Marwari (mwr), Bundeli (bns)

## Architecture

### Three Translation Systems

**1. Legacy Tiered System (`src/`)**
- `src/tiered_orchestrator.py`: Implements 70/20/10 strategy (Hindi pivot / Claude 3.5 / Claude 4.5)
- `src/translators/`: IndicTrans2, NLLB, Hindi pivot implementations
- `src/api_clients/`: Claude API client with cost tracking
- `src/quality/`: BLEU, chrF, TER metrics
- `src/utils/`: Config, logger, cost tracker

**2. Universal Framework (`universal_translate/`)**
- `universal_translate/core/`: Abstract `BaseTranslator` interface and request/response models
- `universal_translate/providers/`: Provider implementations (OpenAI, Anthropic, Gemini, Vertex, Fine-tuned)
- `universal_translate/providers/registry.py`: Dynamic provider registration and discovery
- `universal_translate/prompts/`: Multi-language prompt templates with examples
- `universal_translate/processors/`: Input/output format handlers (CoNLL-U, plain text)

**3. MT Training System (`src/training/`)**
- `src/training/config/`: Comprehensive configuration system (training, LoRA, model, data, merging, ensemble)
- `src/training/models/`: Model wrappers for NLLB, mBART, mT5, IndicTrans2, Llama, Mistral with registry pattern
- `src/training/trainers/`: LoRA and causal LM trainers with HuggingFace Trainer integration
- `src/training/data/`: Parallel corpus loaders, synthetic data creation, preprocessing
- `src/training/merging/`: Model merging methods (TIES, DARE, Model Soups)
- `src/training/ensemble/`: Ensemble translation (COMET, MBR, Voting)
- `src/training/pipeline/`: Hybrid translation pipeline with quality-based routing

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

### MT Model Training and Fine-Tuning

**Train Models with LoRA Fine-Tuning**:
```bash
# Train NLLB model with LoRA
python scripts/train_model.py \
  --model nllb-600m \
  --source-lang hi \
  --target-lang bho \
  --train-data data/hi_bho_train.tsv \
  --output models/nllb-bhojpuri \
  --epochs 10 \
  --batch-size 16

# Train with validation data and custom LoRA parameters
python scripts/train_model.py \
  --model nllb-1.3b \
  --source-lang hi \
  --target-lang mwr \
  --train-data data/train.tsv \
  --val-data data/val.tsv \
  --output models/nllb-marwari \
  --lora-r 32 \
  --lora-alpha 64 \
  --learning-rate 5e-4

# Train causal LM (Llama, Mistral)
python scripts/train_model.py \
  --model llama3-8b \
  --source-lang en \
  --target-lang hi \
  --train-data data/train.json \
  --output models/llama-hindi \
  --epochs 5 \
  --batch-size 8 \
  --gradient-accumulation 8

# Train with YAML config file
python scripts/train_model.py \
  --config config/training/my_config.yaml

# Resume from checkpoint
python scripts/train_model.py \
  --model nllb-600m \
  --source-lang hi \
  --target-lang bho \
  --train-data data/train.tsv \
  --output models/nllb-bho \
  --resume-from-checkpoint models/nllb-bho/checkpoint-1000
```

**Merge Multiple Fine-Tuned Models**:
```bash
# Merge with TIES (Task Inference and Edit Selection)
python scripts/merge_models.py \
  --method ties \
  --base-model facebook/nllb-200-distilled-600M \
  --models models/nllb-bho models/nllb-mag models/nllb-awa \
  --output models/nllb-merged-ties \
  --density 0.2

# Merge with DARE (Drop And REscale)
python scripts/merge_models.py \
  --method dare \
  --base-model facebook/nllb-200-distilled-600M \
  --models models/nllb-bho models/nllb-mag \
  --output models/nllb-merged-dare \
  --drop-rate 0.95

# Create Model Soup (uniform averaging)
python scripts/merge_models.py \
  --method model_soups \
  --base-model facebook/nllb-200-distilled-600M \
  --models models/nllb-bho/checkpoint-*/ \
  --output models/nllb-soup \
  --soup-type uniform

# Create Model Soup (greedy selection with evaluation)
python scripts/merge_models.py \
  --method model_soups \
  --base-model facebook/nllb-200-distilled-600M \
  --models models/nllb-bho/checkpoint-*/ \
  --output models/nllb-soup-greedy \
  --soup-type greedy \
  --eval-data data/val.tsv
```

**Hybrid Translation (Fine-Tuned + API + Ensemble)**:
```bash
# Basic hybrid translation with fine-tuned model only
python scripts/hybrid_translate.py \
  --finetuned-model models/nllb-bhojpuri \
  --source-lang hi \
  --target-lang bho \
  --input texts.txt \
  --output translations.txt

# With API fallback for low-confidence translations
python scripts/hybrid_translate.py \
  --finetuned-model models/nllb-bhojpuri \
  --api-provider openai \
  --api-model gpt-4o-mini \
  --source-lang hi \
  --target-lang bho \
  --input texts.txt \
  --output translations.txt \
  --quality-threshold 0.7 \
  --finetuned-percentage 90.0 \
  --api-percentage 10.0

# With ensemble for critical translations
python scripts/hybrid_translate.py \
  --finetuned-model models/nllb-bho \
  --ensemble-models models/nllb-bho models/mbart-bho models/mt5-bho \
  --ensemble-method comet \
  --source-lang hi \
  --target-lang bho \
  --input texts.txt \
  --output translations.txt \
  --finetuned-percentage 90.0 \
  --api-percentage 9.0 \
  --ensemble-percentage 1.0

# With COMET-QE quality estimation (more accurate but slower)
python scripts/hybrid_translate.py \
  --finetuned-model models/nllb-bho \
  --api-provider anthropic \
  --api-model claude-haiku-4.5 \
  --use-comet-qe \
  --source-lang hi \
  --target-lang bho \
  --text "नमस्ते, आप कैसे हैं?"

# Using fine-tuned model with adapter only
python scripts/hybrid_translate.py \
  --finetuned-model models/nllb-bho-adapter \
  --base-model facebook/nllb-200-distilled-600M \
  --adapter-only \
  --source-lang hi \
  --target-lang bho \
  --input texts.txt \
  --output translations.txt
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

### MT Training API

```python
from src.training.config import TrainingConfig, LoRAConfig, ModelConfig, DataConfig
from src.training.trainers import LoRATrainer
from src.training.models import ModelFactory
from src.training.data import ParallelCorpusLoader

# Load training data
loader = ParallelCorpusLoader(
    train_file="data/hi_bho_train.tsv",
    val_file="data/hi_bho_val.tsv",
    format="tsv"
)
dataset_dict = loader.load()

# Create model
model_config = ModelConfig.create_preset(
    "nllb-600m",
    source_lang="hi",
    target_lang="bho"
)
model = ModelFactory.create_from_config(model_config)

# Configure LoRA and training
lora_config = LoRAConfig(r=16, lora_alpha=32)
training_config = TrainingConfig(
    output_dir="models/nllb-bho",
    num_train_epochs=10,
    per_device_train_batch_size=16
)

# Prepare model for training
model.prepare_for_training(lora_config, training_config)

# Train
trainer = LoRATrainer(
    model=model,
    train_dataset=dataset_dict['train'],
    eval_dataset=dataset_dict['validation'],
    training_config=training_config
)
metrics = trainer.train()

# Save model
model.save("models/nllb-bho", save_full_model=False)
```

**Model Merging API**:
```python
from src.training.merging import TIESMerger, DAREMerger, ModelSoups
from src.training.merging.base import MergeConfig

# TIES merging
config = MergeConfig(method="ties", density=0.2)
merger = TIESMerger(
    base_model_path="facebook/nllb-200-distilled-600M",
    config=config
)
result = merger.merge(
    model_paths=["models/nllb-bho", "models/nllb-mag"],
    output_path="models/nllb-merged"
)
```

**Hybrid Pipeline API**:
```python
from src.training.pipeline import HybridPipeline, HybridConfig
from universal_translate.providers import FineTunedProvider, OpenAIProvider

# Create providers
finetuned = FineTunedProvider(
    model_path="models/nllb-bho",
    device="cuda"
)
finetuned.initialize()

api = OpenAIProvider(model='gpt-4o-mini')

# Configure hybrid pipeline
config = HybridConfig(
    finetuned_percentage=90.0,
    api_percentage=10.0,
    quality_threshold=0.7
)

# Create pipeline
pipeline = HybridPipeline(
    finetuned_model=finetuned,
    api_model=api,
    config=config
)

# Translate
result = pipeline.translate(
    texts=["नमस्ते", "आप कैसे हैं?"],
    source_lang="hi",
    target_lang="bho"
)

print(result.translations)
print(result.get_route_statistics())
print(f"Cost: ${result.costs['total']}")
```

**Fine-Tuned Model Provider API**:
```python
from universal_translate.providers import FineTunedProvider, create_from_training_run
from universal_translate.core import TranslationRequest

# Create provider from training output
provider = create_from_training_run(
    training_output_dir="models/nllb-bho",
    adapter_only=True,
    device="cuda"
)

# Or create directly
provider = FineTunedProvider(
    model_path="models/nllb-bho",
    base_model="facebook/nllb-200-distilled-600M",
    adapter_only=True,
    device="cuda"
)
provider.initialize()

# Translate
request = TranslationRequest(
    units=["नमस्ते", "आप कैसे हैं?"],
    source_lang="hi",
    target_lang="bho"
)
response = provider.translate_sync(request)
print(response.results[0].translation)
print(f"Cost: ${response.cost_info.total_cost}")  # Always $0 for local models
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
├── src/                          # Legacy and training systems
│   ├── tiered_orchestrator.py   # 70/20/10 strategy orchestrator
│   ├── translators/              # IndicTrans2, NLLB, Hindi pivot
│   ├── api_clients/              # Claude API client
│   ├── quality/                  # Quality metrics (BLEU, chrF, TER)
│   ├── utils/                    # Config, logger, cost tracker
│   └── training/                 # MT training infrastructure (NEW)
│       ├── config/               # Training, LoRA, model, data configs
│       ├── models/               # Model wrappers + registry
│       ├── trainers/             # LoRA and causal LM trainers
│       ├── data/                 # Data loaders, synthetic data
│       ├── merging/              # TIES, DARE, Model Soups
│       ├── ensemble/             # COMET, MBR, Voting
│       └── pipeline/             # Hybrid translation pipeline
├── universal_translate/          # Modern multi-provider framework
│   ├── core/                     # Abstract interfaces (BaseTranslator)
│   ├── providers/                # OpenAI, Anthropic, Gemini, Vertex, FineTuned
│   ├── prompts/                  # Language-specific prompt templates
│   ├── processors/               # Format handlers (CoNLL-U, text)
│   └── config/prompts/           # YAML prompt configurations
├── scripts/
│   ├── translate_cli.py          # Universal CLI (RECOMMENDED)
│   ├── translate.py              # Legacy CLI with tiered strategy
│   ├── translate_corpus.py       # Directory structure preservation
│   ├── train_model.py            # Train models with LoRA (NEW)
│   ├── merge_models.py           # Merge fine-tuned models (NEW)
│   ├── hybrid_translate.py       # Hybrid translation pipeline (NEW)
│   ├── download.py               # Download datasets/models
│   └── verify.py                 # Verify installation
├── examples/                     # Python usage examples
├── docs/                         # Documentation
│   └── MT_TRAINING_INTEGRATION.md  # MT training design document
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

### MT Training + Hybrid Pipeline (Best Cost/Quality)

The hybrid approach combines fine-tuned models with API fallback for optimal cost/quality:

- **Fine-tuned models (90%)**: Zero cost, trained once on parallel data
- **API refinement (9%)**: Used only for low-confidence translations
- **Ensemble methods (1%)**: For critical translations requiring highest quality

**Cost comparison for 12M translation pairs**:
- Pure API (OpenAI GPT-4o-mini): ~$740
- Pure API (Anthropic Claude Haiku): ~$3,600
- Hybrid (90/9/1 with fine-tuned NLLB): ~$74 (10x reduction)
- Training cost (one-time): ~$20 GPU hours on V100

**Expected quality improvements**:
- Fine-tuned model: +3 to +8 BLEU over base model
- TIES/DARE merging: +1.7 to +1.9 BLEU additional
- Ensemble methods: +0.5 to +1.5 BLEU additional
- Total improvement: +5 to +11 BLEU vs base model

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

### Training Your Own MT Models

**Quick Start**:
1. Prepare parallel corpus (TSV, CSV, JSON, JSONL, or HuggingFace dataset)
2. Train with LoRA: `python scripts/train_model.py --model nllb-600m --source-lang hi --target-lang bho --train-data data/train.tsv --output models/my-model`
3. Use in hybrid pipeline: `python scripts/hybrid_translate.py --finetuned-model models/my-model --source-lang hi --target-lang bho --text "नमस्ते"`

**Supported Models**:
- **Seq2Seq**: NLLB (600M, 1.3B, 3.3B), mBART (large), mT5 (base, large), IndicTrans2 (200M, 1B)
- **Causal LMs**: Llama 3 (8B), Mistral (7B) - requires instruction formatting

**Training Configuration**:
- All configs support YAML files for complex setups
- LoRA parameters: `r` (rank, default 16), `alpha` (scaling, default 32), `dropout` (default 0.1)
- Training: epochs, batch size, learning rate, warmup, gradient accumulation
- Auto-optimization for V100 GPUs (batch size 16, gradient accumulation 4)

**Model Merging Strategies**:
1. **TIES-Merging**: Best for models trained on related tasks (e.g., multiple dialects)
   - Trims low-magnitude parameters, elects majority sign, merges aligned parameters
   - Expected: +1.7 BLEU over single best model
2. **DARE**: Best when models have complementary knowledge
   - Drops 90-99% of parameters, rescales remaining to preserve magnitude
   - Expected: +1.9 BLEU over single best model
3. **Model Soups**: Best for checkpoints from same training run
   - Uniform: Simple average of all checkpoints
   - Greedy: Iteratively add checkpoints that improve validation performance
   - Expected: +0.6-0.8 BLEU over single checkpoint

**Hybrid Pipeline Configuration**:
- `quality_threshold`: Translations below this use API (default 0.7)
- `ensemble_threshold`: Translations below this use ensemble (default 0.5)
- `use_comet_qe`: Enable COMET neural quality estimation (slower but more accurate)
- `force_percentages`: Enforce exact percentages vs quality-based routing

**Data Requirements**:
- Minimum: ~10K parallel pairs for fine-tuning
- Recommended: 100K+ pairs for production quality
- Can bootstrap from API translations if no seed corpus available

## Data Sources

- **Samanantar**: 49.7M English-Indic parallel pairs (`ai4bharat/samanantar`)
- **IndicTrans2 Models**: `ai4bharat/indictrans2-*` (1B and 200M variants)
- **NLLB Models**: `facebook/nllb-200-*` (600M, 1.3B, 3.3B variants)

Download via: `python scripts/download.py`

## References

**Translation Models**:
- **IndicTrans2**: https://arxiv.org/abs/2305.16307
- **Samanantar**: https://arxiv.org/abs/2104.05596
- **NLLB**: https://arxiv.org/abs/2207.04672

**Fine-Tuning and Merging**:
- **LoRA (Low-Rank Adaptation)**: https://arxiv.org/abs/2106.09685
- **TIES-Merging**: https://arxiv.org/abs/2306.01708
- **DARE (Drop And REscale)**: https://arxiv.org/abs/2311.03099
- **Model Soups**: https://arxiv.org/abs/2203.05482

**Quality Estimation and Ensemble**:
- **COMET**: https://arxiv.org/abs/2009.09025
- **MBR Decoding**: https://aclanthology.org/N04-1022/
- **Quality-Aware Decoding**: https://arxiv.org/abs/2205.00978

**API Providers**:
- **OpenAI Prompt Caching**: https://platform.openai.com/docs/guides/prompt-caching
- **Anthropic Prompt Caching**: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
- **Gemini Context Caching**: https://ai.google.dev/gemini-api/docs/caching
