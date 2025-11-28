# Usage Guide

Comprehensive guide for using BhaashikLowsResLangTrans.

## Table of Contents

1. [Basic Translation](#basic-translation)
2. [Tiered Translation Strategy](#tiered-translation-strategy)
3. [Cost Estimation](#cost-estimation)
4. [Batch Translation](#batch-translation)
5. [Quality Assessment](#quality-assessment)
6. [API Usage](#api-usage)
7. [Advanced Examples](#advanced-examples)

## Basic Translation

### Using IndicTrans2 (Free)

Best for 22 supported Indic languages.

```bash
# Single text
python scripts/translate.py \
  --src en \
  --tgt hi \
  --text "Hello, how are you?" \
  --indictrans2

# File input/output
python scripts/translate.py \
  --src en \
  --tgt bn \
  --input source.txt \
  --output translation.txt \
  --indictrans2
```

Supported languages: as, bn, gu, hi, kn, ks, kok, ml, mni, mr, ne, or, pa, sa, sd, ta, te, ur, brx, sat, mai, doi

### Using NLLB (Free)

Alternative model with broader coverage.

```bash
python scripts/translate.py \
  --src en \
  --tgt bho \
  --input source.txt \
  --output translation.txt \
  --nllb
```

### Using Hindi Pivot (Free)

For languages linguistically close to Hindi (Bhojpuri, Magahi, etc.).

```bash
python scripts/translate.py \
  --src en \
  --tgt bho \
  --input source.txt \
  --output translation.txt \
  --hindi-pivot
```

## Tiered Translation Strategy

The tiered strategy optimally combines free and paid translation methods.

### Default Strategy (70/20/10)

- 70% using free Hindi pivot
- 20% using Claude Haiku 3.5 (enhanced quality)
- 10% using Claude Haiku 4.5 (premium quality)

```bash
python scripts/translate.py \
  --src en \
  --tgt bho \
  --input source.txt \
  --output translation.txt \
  --tiered
```

### Custom Tier Distribution

```bash
# 90% free, 8% enhancement, 2% premium
python scripts/translate.py \
  --src en \
  --tgt bho \
  --input source.txt \
  --output translation.txt \
  --tiered \
  --tier-free 90 \
  --tier-enhancement 8 \
  --tier-premium 2
```

### Pure Free Strategy

```bash
# 100% free Hindi pivot
python scripts/translate.py \
  --src en \
  --tgt bho \
  --input source.txt \
  --output translation.txt \
  --tiered \
  --tier-free 100 \
  --tier-enhancement 0 \
  --tier-premium 0
```

## Cost Estimation

Always estimate costs before translating with APIs!

### Estimate Cost

```bash
# Estimate for 10,000 samples
python scripts/translate.py \
  --src en \
  --tgt bho \
  --num-samples 10000 \
  --estimate-only

# With custom tier distribution
python scripts/translate.py \
  --src en \
  --tgt bho \
  --num-samples 10000 \
  --tier-free 80 \
  --tier-enhancement 15 \
  --tier-premium 5 \
  --estimate-only
```

### Cost for Different Languages

```bash
# All 6 unsupported languages
for lang in bho mag awa bra mwr bns; do
  echo "Estimating cost for $lang"
  python scripts/translate.py \
    --src en \
    --tgt $lang \
    --num-samples 2000000 \
    --estimate-only
done
```

## Batch Translation

### Translate Files

```bash
# Prepare input file (one sentence per line)
cat > input.txt <<EOF
Hello, how are you?
What is your name?
I am learning a new language.
EOF

# Translate
python scripts/translate.py \
  --src en \
  --tgt bho \
  --input input.txt \
  --output output.txt \
  --tiered \
  --batch-size 32
```

### Translate Large Files

```bash
# For very large files, process in chunks
split -l 10000 large_input.txt chunk_

# Translate each chunk
for file in chunk_*; do
  python scripts/translate.py \
    --src en \
    --tgt bho \
    --input "$file" \
    --output "translated_$file" \
    --tiered
done

# Combine results
cat translated_chunk_* > final_translation.txt
```

## Quality Assessment

### Compare Translation Methods

```python
# Use Python API for advanced usage
from src.tiered_orchestrator import TieredOrchestrator
from src.translators import IndicTrans2Translator, NLLBTranslator
from src.quality import QualityMetrics

# Load texts and references
texts = [...]  # Your source texts
references = [...]  # Reference translations

# Translate with different methods
indictrans2 = IndicTrans2Translator()
nllb = NLLBTranslator()

trans_it = indictrans2.translate(texts, 'en', 'hi')
trans_nllb = nllb.translate(texts, 'en', 'hi')

# Calculate quality metrics
metrics = QualityMetrics()
results = metrics.compare_translations(
    {
        'IndicTrans2': trans_it,
        'NLLB': trans_nllb
    },
    references
)

# Print comparison
for system, scores in results.items():
    print(f"\n{system}:")
    print(f"  BLEU: {scores['corpus_bleu']:.2f}")
    print(f"  chrF: {scores['chrf']:.2f}")
```

## API Usage

### Python API Examples

#### Simple Translation

```python
from src.translators import IndicTrans2Translator

# Initialize translator
translator = IndicTrans2Translator()

# Translate single text
result = translator.translate(
    "Hello, how are you?",
    src_lang='en',
    tgt_lang='hi'
)
print(result)

# Translate batch
texts = ["Hello", "Good morning", "Thank you"]
results = translator.translate_batch(
    texts,
    src_lang='en',
    tgt_lang='hi'
)
```

#### Tiered Translation

```python
from src.tiered_orchestrator import TieredOrchestrator

# Initialize orchestrator
orchestrator = TieredOrchestrator()

# Get cost estimate
estimate = orchestrator.get_cost_estimate(
    num_samples=10000,
    avg_tokens_per_sample=100
)
print(f"Estimated cost: {estimate['currency']} {estimate['total_cost']:.2f}")

# Translate
texts = [...]  # Your texts
results = orchestrator.translate(
    texts,
    src_lang='en',
    tgt_lang='bho'
)

# Access results
translations = results['translations']
tier_assignments = results['tier_assignments']
cost_summary = results['cost_summary']
```

#### Dataset Translation

```python
from datasets import load_dataset
from src.tiered_orchestrator import TieredOrchestrator

# Load dataset
dataset = load_dataset('ai4bharat/samanantar', 'hi', split='train')

# Translate first 1000 samples
orchestrator = TieredOrchestrator()
results = orchestrator.translate_dataset(
    dataset,
    src_lang='hi',
    tgt_lang='bho',
    src_field='tgt',  # Hindi text
    max_samples=1000
)

# Quality metrics automatically calculated if references available
print(results['quality_metrics'])
```

## Advanced Examples

### 1. Custom Configuration

```python
from src.utils.config import Config
from src.tiered_orchestrator import TieredOrchestrator

# Load custom config
config = Config('path/to/custom_config.yaml')

# Custom tier percentages
orchestrator = TieredOrchestrator(
    config=config,
    tier_percentages={
        'free': 85,
        'enhancement': 10,
        'premium': 5
    }
)

# Translate
results = orchestrator.translate(texts, 'en', 'bho')
```

### 2. Cost-Aware Translation

```python
from src.tiered_orchestrator import TieredOrchestrator
from src.utils.cost_tracker import CostTracker

# Initialize with budget
cost_tracker = CostTracker(
    log_file='logs/cost.json',
    currency='INR'
)

orchestrator = TieredOrchestrator(cost_tracker=cost_tracker)

# Check budget before translating
budget_status = cost_tracker.check_budget(
    daily_budget=5000,  # INR
    total_budget=100000
)

if budget_status['within_total_budget']:
    results = orchestrator.translate(texts, 'en', 'bho')
else:
    print("Budget exceeded!")
```

### 3. Quality-Driven Tier Assignment

```python
from src.tiered_orchestrator import TieredOrchestrator
from src.translators import HindiPivotTranslator
from src.quality import QualityMetrics

# First pass: free translation
pivot = HindiPivotTranslator()
free_translations = pivot.translate(texts, 'en', 'bho')

# Assess quality
metrics = QualityMetrics()
# Compare with references or use other heuristics

# Second pass: enhance low-quality translations with API
# (Implementation depends on your quality criteria)
```

### 4. Parallel Processing

```python
from multiprocessing import Pool
from src.translators import IndicTrans2Translator

def translate_chunk(chunk):
    translator = IndicTrans2Translator()
    return translator.translate(chunk, 'en', 'hi')

# Split data into chunks
chunks = [texts[i:i+1000] for i in range(0, len(texts), 1000)]

# Process in parallel
with Pool(4) as p:
    results = p.map(translate_chunk, chunks)

# Flatten results
all_translations = [t for chunk_results in results for t in chunk_results]
```

## Monitoring and Logs

### View Cost Tracking

```bash
# View cost summary
python -c "
from src.utils.cost_tracker import CostTracker
tracker = CostTracker('$BASE_DIR/logs/cost_tracking.json')
tracker.print_summary()
"
```

### View Translation Logs

```bash
# View logs
tail -f $BASE_DIR/logs/translation.log

# View cost log
cat $BASE_DIR/logs/cost_tracking.json | jq .
```

## Best Practices

1. **Always estimate costs first** when using API-based translation
2. **Start with free methods** (IndicTrans2, Hindi pivot) and upgrade selectively
3. **Use appropriate batch sizes** based on GPU memory
4. **Monitor API usage** to avoid unexpected costs
5. **Save intermediate results** for long-running translations
6. **Validate quality** with sample reviews
7. **Use version control** for configuration changes

## Tips for Cost Optimization

1. Use 100% free tier for initial experiments
2. Gradually increase API usage based on quality needs
3. Use Hindi pivot for languages linguistically close to Hindi
4. Batch API calls to reduce overhead
5. Enable prompt caching when available
6. Monitor daily/monthly budgets

## Getting Help

- Check logs: `$BASE_DIR/logs/translation.log`
- Run verification: `python scripts/verify.py --all`
- Review configuration: `cat config/config.yaml`
- Check cost tracking: `cat $BASE_DIR/logs/cost_tracking.json`
