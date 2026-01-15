# MT Training Integration Design

**Integration of BhaashikMTTrain capabilities into BhaashikLowsResLangTrans**

## Overview

This document outlines the architecture for integrating neural MT training, fine-tuning, model merging, and ensemble capabilities from BhaashikMTTrain into the existing BhaashikLowsResLangTrans project.

## Design Philosophy

### Non-Disruptive Integration
- **Preserve existing systems**: Keep both legacy (`src/`) and universal (`universal_translate/`) translation systems
- **Add, don't replace**: Training capabilities are additive features
- **Backward compatible**: Existing scripts and workflows continue to work

### Hybrid Approach
- **Combine strengths**: API quality + local model efficiency + fine-tuned specialization
- **Cost optimization**: Use API to bootstrap training data, then fine-tune local models
- **Quality improvement**: Ensemble and merge multiple approaches

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                BhaashikLowsResLangTrans                      │
│                                                              │
│  ┌────────────────┐        ┌──────────────────┐            │
│  │  Existing API  │        │  NEW: Training   │            │
│  │  Translation   │◄──────►│  Infrastructure  │            │
│  │  (Universal)   │        │                  │            │
│  └────────────────┘        └──────────────────┘            │
│         │                          │                         │
│         │                          │                         │
│         ▼                          ▼                         │
│  ┌────────────────────────────────────────────┐            │
│  │        Hybrid Translation Pipeline         │            │
│  │  • API (high quality, costly)              │            │
│  │  • Fine-tuned models (specialized, cheap)  │            │
│  │  • Pre-trained models (baseline)           │            │
│  │  • Merged models (multi-run combination)   │            │
│  │  • Ensemble (multi-model consensus)        │            │
│  └────────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────┘
```

## Module Structure

```
src/
├── training/                      # NEW: Training infrastructure
│   ├── __init__.py
│   ├── trainers/
│   │   ├── __init__.py
│   │   ├── base.py               # Abstract trainer interface
│   │   ├── lora_trainer.py       # LoRA fine-tuning for seq2seq
│   │   ├── causal_lm_trainer.py  # LLM fine-tuning
│   │   └── metrics.py            # BLEU, ChrF evaluation
│   ├── models/
│   │   ├── __init__.py
│   │   ├── factory.py            # Model creation from config
│   │   ├── registry.py           # Model registration system
│   │   ├── hf_seq2seq.py         # NLLB, mBART, mT5 wrappers
│   │   ├── hf_llm.py             # Llama, Mistral wrappers
│   │   └── indictrans2.py        # IndicTrans2 integration
│   ├── merging/
│   │   ├── __init__.py
│   │   ├── ties.py               # TIES-Merging (conflict resolution)
│   │   ├── dare.py               # DARE (drop and rescale)
│   │   ├── model_soups.py        # Checkpoint averaging
│   │   └── lora_merging.py       # LoRA adapter combination
│   ├── ensemble/
│   │   ├── __init__.py
│   │   ├── comet.py              # COMET-based selection
│   │   ├── mbr.py                # Minimum Bayes Risk
│   │   └── voting.py             # Simple voting
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loaders.py            # Parallel corpus loading
│   │   ├── processors.py         # Preprocessing pipelines
│   │   └── synthetic.py          # Convert API output to training data
│   └── config/
│       ├── __init__.py
│       ├── training_config.py    # Training hyperparameters
│       └── model_config.py       # Model configurations

config/
├── training/                      # NEW: Training configs
│   ├── lora_default.yaml
│   ├── llm_finetuning.yaml
│   └── v100_optimized.yaml
└── models/                        # NEW: Model configs
    ├── nllb_600m.yaml
    ├── llama3_8b.yaml
    └── mistral_7b.yaml

scripts/
├── train_model.py                # NEW: Training CLI
├── merge_models.py               # NEW: Model merging CLI
├── create_training_data.py       # NEW: Bootstrap from API
└── hybrid_translate.py           # NEW: Multi-model pipeline

examples/
├── example_6_model_training.py   # NEW: Training example
├── example_7_model_merging.py    # NEW: Merging example
└── example_8_hybrid_pipeline.py  # NEW: Hybrid translation
```

## Integration Scenarios

### Scenario 1: No Seed Corpus (Bootstrap from API)

**Workflow:**
1. **Generate synthetic corpus**: Use API translation on Hindi data
2. **Create training pairs**: Hindi → API_Bhojpuri pairs
3. **Fine-tune baseline model**: NLLB-600M + LoRA on synthetic data
4. **Iterative improvement**:
   - Translate test set with fine-tuned model
   - Use API to correct/improve outputs
   - Add high-quality pairs to training data
   - Fine-tune again

**Example:**
```bash
# Step 1: Create synthetic training data
python scripts/create_training_data.py \
  --provider anthropic \
  --source-lang hi \
  --target-lang bho \
  --input-file input/hindi_monolingual.txt \
  --output-pairs data/synthetic/hi-bho-pairs.tsv \
  --num-samples 10000

# Step 2: Fine-tune NLLB on synthetic data
python scripts/train_model.py \
  --model nllb-600m \
  --data data/synthetic/hi-bho-pairs.tsv \
  --source hi \
  --target bho \
  --use-lora \
  --epochs 5 \
  --output models/nllb-hi-bho-synthetic

# Step 3: Hybrid translation (fine-tuned + API for difficult cases)
python scripts/hybrid_translate.py \
  --base-model models/nllb-hi-bho-synthetic \
  --api-provider anthropic \
  --difficulty-threshold 0.8 \
  --input test.txt \
  --output translations.txt
```

### Scenario 2: With Seed Corpus

**Workflow:**
1. **Direct fine-tuning**: Use existing parallel corpus
2. **Multi-run training**: Train with different hyperparameters
3. **Model merging**: Combine checkpoints with TIES-Merging
4. **Quality enhancement**: Use API for low-confidence translations
5. **Ensemble**: Combine merged model + API for final output

**Example:**
```bash
# Step 1: Fine-tune multiple runs
for seed in 42 123 456; do
  python scripts/train_model.py \
    --model nllb-600m \
    --data data/parallel/hi-bho-corpus.tsv \
    --source hi --target bho \
    --seed $seed \
    --output models/nllb-hi-bho-seed-$seed
done

# Step 2: Merge models with TIES
python scripts/merge_models.py \
  --method ties \
  --models models/nllb-hi-bho-seed-* \
  --output models/nllb-hi-bho-merged \
  --density 0.8

# Step 3: Ensemble with API
python scripts/hybrid_translate.py \
  --base-model models/nllb-hi-bho-merged \
  --api-provider anthropic \
  --ensemble-method comet \
  --input test.txt \
  --output translations.txt
```

### Scenario 3: Experimental Comparison

**Workflow:**
1. Train multiple model types (NLLB, mT5, Llama)
2. Ensemble with different methods (COMET, MBR, Voting)
3. Compare against API baseline
4. Analyze cost-quality tradeoffs

**Example:**
```bash
# Train different models
python scripts/train_model.py --model nllb-600m --data corpus.tsv --output models/nllb
python scripts/train_model.py --model mt5-base --data corpus.tsv --output models/mt5
python scripts/train_model.py --model llama3-8b --data corpus.tsv --output models/llama

# Compare all approaches
python scripts/compare_models.py \
  --models models/nllb models/mt5 models/llama \
  --api-providers openai anthropic gemini \
  --test-data test.tsv \
  --output-report comparison_report.json
```

## Hybrid Translation Pipeline

### Architecture

```python
class HybridTranslator:
    """Combines multiple translation approaches intelligently."""

    def __init__(self, config):
        self.fine_tuned_model = load_model(config.model_path)
        self.api_translator = APITranslator(config.api_provider)
        self.ensemble = EnsembleTranslator(config.ensemble_method)
        self.quality_estimator = QualityEstimator()

    def translate(self, text):
        # 1. Try fine-tuned model first (fast, cheap)
        translation = self.fine_tuned_model.translate(text)

        # 2. Estimate quality
        quality_score = self.quality_estimator.score(text, translation)

        # 3. If quality below threshold, use API
        if quality_score < self.config.quality_threshold:
            translation = self.api_translator.translate(text)

        # 4. Optional: Ensemble multiple approaches
        if self.config.use_ensemble:
            candidates = [
                self.fine_tuned_model.translate(text),
                self.api_translator.translate(text)
            ]
            translation = self.ensemble.select_best(candidates)

        return translation
```

### Cost-Quality Optimization

**Strategy tiers:**
1. **90% fine-tuned model** (cheap, ~$0)
2. **9% API for low-confidence** (moderate cost)
3. **1% ensemble for critical content** (high cost, best quality)

**Estimated savings:**
- Pure API: $740 for 12M pairs
- Hybrid (90/9/1): $74 for 12M pairs (10x cheaper)
- Quality: 95-98% of pure API quality

## Training Configuration

### Default LoRA Settings

```yaml
# config/training/lora_default.yaml
lora:
  rank: 16
  alpha: 32
  dropout: 0.1
  target_modules: [q_proj, v_proj, k_proj, out_proj]

training:
  epochs: 10
  batch_size: 16
  learning_rate: 3e-4
  warmup_steps: 300
  evaluation_strategy: steps
  eval_steps: 500
  save_strategy: steps
  save_steps: 1000
  early_stopping_patience: 3

generation:
  num_beams: 4
  max_length: 256
```

### Model-Specific Configs

**NLLB-600M:**
```yaml
model:
  name: facebook/nllb-200-distilled-600M
  type: seq2seq
  memory: 16GB
  supported_languages: 200+
```

**Llama-3-8B:**
```yaml
model:
  name: meta-llama/Meta-Llama-3-8B-Instruct
  type: causal_lm
  memory: 16GB  # with 8-bit quantization
  instruction_format: llama3
  use_8bit_loading: true
```

## Data Preparation

### From API Translations

```python
class SyntheticDataCreator:
    """Convert API translations to training data."""

    def create_training_pairs(
        self,
        source_texts: List[str],
        api_provider: str,
        source_lang: str,
        target_lang: str
    ) -> List[Tuple[str, str]]:
        """
        Generate synthetic parallel corpus.

        Returns:
            List of (source, target) pairs
        """
        translator = get_api_translator(api_provider)
        pairs = []

        for text in tqdm(source_texts):
            translation = translator.translate(
                text,
                source_lang=source_lang,
                target_lang=target_lang
            )
            pairs.append((text, translation))

        return pairs

    def filter_quality(
        self,
        pairs: List[Tuple[str, str]],
        min_confidence: float = 0.7
    ) -> List[Tuple[str, str]]:
        """Remove low-quality pairs."""
        quality_estimator = QualityEstimator()

        filtered = []
        for src, tgt in pairs:
            score = quality_estimator.score(src, tgt)
            if score >= min_confidence:
                filtered.append((src, tgt))

        return filtered
```

### From Existing Corpus

```python
from src.training.data import ParallelCorpusLoader

loader = ParallelCorpusLoader(
    data_path="data/parallel/hi-bho-corpus.tsv",
    source_lang="hi",
    target_lang="bho",
    format="tsv",
    split_ratios=(0.8, 0.1, 0.1)  # train/eval/test
)

dataset = loader.load()
# Returns: {'train': Dataset, 'eval': Dataset, 'test': Dataset}
```

## Model Merging Integration

### TIES-Merging (Best for conflicting models)

```python
from src.training.merging import TIESMerger

merger = TIESMerger(density=0.8)  # Keep top 80% parameters

merged = merger.merge(
    model_paths=[
        "models/run1",
        "models/run2",
        "models/run3"
    ],
    base_model="facebook/nllb-200-distilled-600M"
)

merged.save("models/merged-ties")
```

### DARE (Drop And REscale)

```python
from src.training.merging import DAREMerger

merger = DAREMerger(
    drop_rate=0.9,  # Drop 90% of deltas
    rescale_mode="adaptive"
)

merged = merger.merge(
    model_paths=["models/run1", "models/run2"],
    base_model="facebook/nllb-200-distilled-600M"
)
```

### Model Soups (Checkpoint averaging)

```python
from src.training.merging import ModelSoups

soup = ModelSoups.greedy_soup(
    checkpoint_paths=[
        "models/checkpoint-1000",
        "models/checkpoint-2000",
        "models/checkpoint-3000"
    ],
    eval_dataset=eval_data
)

soup.save("models/greedy-soup")
```

## Ensemble Integration

### COMET-Based Selection

```python
from src.training.ensemble import COMETEnsemble

ensemble = COMETEnsemble(
    models=[
        load_model("models/nllb"),
        load_model("models/mt5"),
        get_api_translator("anthropic")
    ]
)

translation = ensemble.translate(
    "नमस्ते, आप कैसे हैं?",
    source_lang="hi",
    target_lang="bho"
)
# Automatically selects highest-quality translation
```

### MBR Decoding

```python
from src.training.ensemble import MBRDecoder

decoder = MBRDecoder(
    models=[model1, model2, model3],
    num_samples=20  # Generate 20 candidates per model
)

translation = decoder.decode("source text")
# Returns: consensus translation with confidence score
```

## Quality Estimation

### COMET Score

```python
from src.training.ensemble import COMETScorer

scorer = COMETScorer()
score = scorer.score(
    source="नमस्ते",
    translation="नमस्कार",
    reference=None  # Reference-free scoring
)
# Returns: 0.0-1.0 quality score
```

### Back-Translation Validation

```python
from src.training.ensemble import BackTranslationValidator

validator = BackTranslationValidator(
    forward_model=hi_to_bho_model,
    backward_model=bho_to_hi_model
)

is_valid = validator.validate(
    source="नमस्ते",
    translation="नमस्कार",
    threshold=0.8
)
```

## Updated Config Structure

### config/config.yaml additions

```yaml
# NEW: Training configuration
training:
  enabled: true
  default_framework: lora  # lora, full_finetuning, causal_lm
  checkpoint_dir: checkpoints/

  # GPU settings
  device: cuda
  mixed_precision: fp16
  gradient_checkpointing: true

  # LoRA defaults
  lora:
    rank: 16
    alpha: 32
    dropout: 0.1

  # Training defaults
  hyperparameters:
    epochs: 10
    batch_size: 16
    learning_rate: 3e-4
    warmup_steps: 300

# NEW: Model registry
models:
  hf_seq2seq:
    nllb_600m: facebook/nllb-200-distilled-600M
    nllb_1_3b: facebook/nllb-200-1.3B
    mbart_large: facebook/mbart-large-50-many-to-many-mmt
    mt5_base: google/mt5-base

  hf_llm:
    llama3_8b: meta-llama/Meta-Llama-3-8B-Instruct
    mistral_7b: mistralai/Mistral-7B-Instruct-v0.2

  local_models:
    indictrans2_200m: ai4bharat/indictrans2-en-indic-dist-200M

# NEW: Hybrid translation strategy
hybrid:
  enabled: false

  pipeline:
    - method: fine_tuned
      percentage: 90
      threshold: 0.7  # Quality threshold

    - method: api_fallback
      percentage: 9
      provider: anthropic
      model: claude-haiku-4.5

    - method: ensemble
      percentage: 1
      models: [fine_tuned, api]
      ensemble_method: comet

# NEW: Model merging
merging:
  method: ties  # ties, dare, model_soups, lora_merging
  ties:
    density: 0.8
  dare:
    drop_rate: 0.9
    rescale_mode: adaptive
```

## CLI Commands

### Training

```bash
# Basic training
python scripts/train_model.py \
  --model nllb-600m \
  --data data/parallel/corpus.tsv \
  --source hi --target bho \
  --output models/nllb-hi-bho

# Advanced training with LoRA
python scripts/train_model.py \
  --model llama3-8b \
  --data data/parallel/corpus.tsv \
  --source hi --target bho \
  --use-lora \
  --lora-rank 16 \
  --lora-alpha 32 \
  --epochs 5 \
  --batch-size 8 \
  --learning-rate 3e-4 \
  --output models/llama3-hi-bho
```

### Merging

```bash
# TIES merging
python scripts/merge_models.py \
  --method ties \
  --models models/run1 models/run2 models/run3 \
  --base-model facebook/nllb-200-distilled-600M \
  --density 0.8 \
  --output models/merged-ties

# DARE merging
python scripts/merge_models.py \
  --method dare \
  --models models/run1 models/run2 \
  --drop-rate 0.9 \
  --output models/merged-dare

# Model soups
python scripts/merge_models.py \
  --method model_soups \
  --checkpoints models/ckpt-* \
  --eval-data data/eval.tsv \
  --output models/soup
```

### Hybrid Translation

```bash
# Hybrid with quality-based routing
python scripts/hybrid_translate.py \
  --base-model models/nllb-hi-bho \
  --api-provider anthropic \
  --quality-threshold 0.8 \
  --input test.txt \
  --output translations.txt

# Multi-model ensemble
python scripts/hybrid_translate.py \
  --models models/nllb models/mt5 models/llama \
  --ensemble-method comet \
  --input test.txt \
  --output translations.txt
```

### Create Training Data

```bash
# Bootstrap from API
python scripts/create_training_data.py \
  --provider anthropic \
  --model claude-haiku-4.5 \
  --source-lang hi \
  --target-lang bho \
  --input-file input/hindi_texts.txt \
  --output-pairs data/synthetic/hi-bho.tsv \
  --num-samples 10000 \
  --quality-filter 0.7
```

## Implementation Phases

### Phase 1: Core Infrastructure (Week 1)
- [ ] Copy core training modules from BhaashikMTTrain
- [ ] Adapt LoRATrainer and CausalLMTrainer
- [ ] Integrate with existing config system
- [ ] Add model registry and factory

### Phase 2: Model Support (Week 2)
- [ ] Add NLLB, mBART, mT5 wrappers
- [ ] Add Llama, Mistral wrappers
- [ ] Integrate IndicTrans2 training
- [ ] Test on small datasets

### Phase 3: Merging & Ensemble (Week 3)
- [ ] Implement TIES-Merging
- [ ] Implement DARE
- [ ] Implement Model Soups
- [ ] Add COMET ensemble
- [ ] Add MBR decoding

### Phase 4: Hybrid Pipeline (Week 4)
- [ ] Create HybridTranslator class
- [ ] Implement quality-based routing
- [ ] Add synthetic data creation
- [ ] Integrate with existing universal_translate

### Phase 5: Testing & Documentation (Week 5)
- [ ] Create example scripts
- [ ] Write comprehensive documentation
- [ ] Test all scenarios (with/without seed corpus)
- [ ] Benchmark cost-quality tradeoffs

## Success Metrics

### Quality Targets
- Fine-tuned model BLEU: 30-35 (baseline)
- Merged model BLEU: +1.5-2.5 over baseline
- Hybrid pipeline BLEU: 38-42 (approaching API quality)

### Cost Targets
- 10x cost reduction vs pure API
- <$100 per language pair for 2M sentences

### Performance Targets
- Training time: <2 hours for 10K pairs on V100
- Inference: >100 sentences/sec on GPU
- Merging: <10 minutes for 3 models

## Next Steps

1. Review this design document
2. Get approval for architecture
3. Begin Phase 1 implementation
4. Create initial training examples
5. Test on Bhojpuri as pilot language
