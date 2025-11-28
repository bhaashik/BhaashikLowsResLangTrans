# BhaashikLowsResLangTrans

**Cost-Optimized Translation System for Low-Resource Indic Languages**

A comprehensive Python framework for creating synthetic parallel corpora for low-resource South Asian languages, with focus on cost-effective translation strategies combining open-source models and commercial APIs.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

## 🎯 Key Features

- **Multiple Translation Strategies**: IndicTrans2, NLLB-200, Hindi Pivot, Claude API
- **Cost-Optimized Tiered Approach**: Intelligently combines free and paid methods (70/20/10 strategy)
- **Comprehensive Cost Tracking**: Monitor and control API spending
- **Quality Assessment**: Built-in BLEU, chrF, and TER metrics
- **22 Supported Languages**: Direct support via IndicTrans2
- **6 Low-Resource Languages**: Hindi pivot + API enhancement (Bhojpuri, Magahi, Awadhi, Braj, Marwari, Bundeli)
- **Batch Processing**: Efficient handling of large-scale translations
- **Flexible Architecture**: Easy to extend and customize

## 📊 Supported Languages

### Directly Supported (FREE via IndicTrans2)
Assamese, Bengali, Gujarati, Hindi, Kannada, Kashmiri, Konkani, Malayalam, Manipuri, Marathi, Nepali, Odia, Punjabi, Sanskrit, Sindhi, Tamil, Telugu, Urdu, Bodo, Santhali, Maithili, Dogri

### Low-Resource (Hindi Pivot + Optional API Enhancement)
- **Bhojpuri** (~51M speakers)
- **Magahi** (~13M speakers)
- **Awadhi** (~38M speakers)
- **Braj** (~1M speakers)
- **Marwari** (~13M speakers)
- **Bundeli** (~3M speakers)

## 🚀 Quick Start

### Installation (5 minutes)

```bash
# 1. Clone repository
git clone https://github.com/bhaashik/BhaashikLowsResLangTrans.git
cd BhaashikLowsResLangTrans

# 2. Create conda environment
conda env create -f environment.yml
conda activate bhaashik-translation

# 3. Install package
pip install -e .

# 4. Configure API key
cp .env.example .env
nano .env  # Add your OPENAI_API_KEY

# 5. Test installation (translates 1 file)
python scripts/translate_hindi_to_lowres_openai.py \
  --target-lang bho \
  --max-files 1
```

**Detailed setup instructions:** See [SETUP.md](SETUP.md)

### Quick Translation

**Translate to Bhojpuri (66,317 sentences):**
```bash
python scripts/translate_hindi_to_lowres_openai.py --target-lang bho
```

**All 3 configured languages:**
```bash
for lang in bho mag mai; do
    python scripts/translate_hindi_to_lowres_openai.py --target-lang $lang
done
```

**Cost:** ~$4 per language (~$12 for all 3) with GPT-4o-mini
**Time:** ~8-10 hours per language

### Usage Documentation

- **[QUICK_START_OPENAI.md](QUICK_START_OPENAI.md)** - Quick reference guide
- **[FULL_TRANSLATION_GUIDE.md](FULL_TRANSLATION_GUIDE.md)** - Complete usage guide
- **[SETUP.md](SETUP.md)** - Detailed setup instructions
- **[TRANSFER_CHECKLIST.md](TRANSFER_CHECKLIST.md)** - Moving to another computer

## 📦 Project Structure

```
BhaashikLowsResLangTrans/
├── environment.yml              # Conda environment specification
├── pyproject.toml              # Python package configuration
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variables template
├── SETUP.md                    # Complete setup guide
├── TRANSFER_CHECKLIST.md       # Migration guide
│
├── universal_translate/        # Core translation package
│   ├── core/                   # Base classes and models
│   ├── providers/              # Translation providers (OpenAI, Anthropic, etc.)
│   ├── prompts/                # Prompt management
│   ├── config/                 # Configuration files
│   │   └── prompts/            # Language-specific prompts
│   └── data/                   # Example data for prompt caching
│       └── examples/           # Monolingual examples
│
├── scripts/                    # Translation scripts
│   ├── translate_hindi_to_lowres_openai.py
│   ├── prepare_monolingual_examples.py
│   └── extract_plaintext_from_conllu.py
│
├── input/                      # Input data
│   └── converted/Hindi/        # Hindi plain text and CoNLL-U
│       ├── plain-text/
│       └── mappings/
│
└── output/                     # Translation output
    ├── Bhojpuri/
    ├── Magahi/
    └── Maithili/
```

## 🔧 Environment Setup

### Prerequisites

- Python 3.10+ (3.12 recommended)
- Conda (Miniconda or Anaconda)
- OpenAI API key
- 50+ GB free disk space

### Files for Environment Setup

| File | Purpose |
|------|---------|
| `environment.yml` | Conda environment with all dependencies |
| `pyproject.toml` | Python package metadata and dependencies |
| `requirements.txt` | Pip-installable dependencies list |
| `.env.example` | Environment variables template |

**To recreate environment on any computer:**
```bash
conda env create -f environment.yml
conda activate bhaashik-translation
pip install -e .
cp .env.example .env  # Then add your API key
```

### Basic Usage

```bash
# Translate English to Hindi (FREE)
python scripts/translate.py \
  --src en \
  --tgt hi \
  --text "Hello, how are you?" \
  --indictrans2

# Translate to low-resource language using Hindi pivot (FREE)
python scripts/translate.py \
  --src en \
  --tgt bho \
  --input texts.txt \
  --output translations.txt \
  --hindi-pivot

# Use tiered strategy for optimal quality/cost (COSTS MONEY)
python scripts/translate.py \
  --src en \
  --tgt bho \
  --input texts.txt \
  --output translations.txt \
  --tiered

# Translate entire corpus with directory structure preservation (NEW!)
python scripts/translate_corpus.py \
  --input-dir /path/to/UD_Hindi \
  --output-dir /path/to/UD_Bhojpuri \
  --src hi \
  --tgt bho \
  --format conllu \
  --hindi-pivot
```

### Python API

```python
from src.translators import IndicTrans2Translator

# Initialize translator
translator = IndicTrans2Translator()

# Translate
translation = translator.translate(
    "Hello, how are you?",
    src_lang='en',
    tgt_lang='hi'
)
print(translation)
```

## 💰 Cost-Optimized Strategy

The tiered approach optimally balances quality and cost:

| Tier | Method | Percentage | Cost (INR/1M tokens) | Use Case |
|------|--------|------------|---------------------|----------|
| 1 | Hindi Pivot (IndicTrans2) | 70% | FREE | Bulk translation |
| 2 | Claude Haiku 3.5 | 20% | 408 | Quality enhancement |
| 3 | Claude Haiku 4.5 | 10% | 510 | Premium quality |

**Estimated costs for 6 languages (12M sentence pairs):**
- Pure free (Hindi pivot): ₹0
- Tiered (70/20/10): ₹62,912 (~$740)
- Per language: ₹10,485 (~$123)

### Cost Estimation

```bash
# Estimate before translating
python scripts/translate.py \
  --src en \
  --tgt bho \
  --num-samples 10000 \
  --estimate-only
```

## 📚 Documentation

- [**SETUP.md**](SETUP.md) - Detailed installation and configuration
- [**USAGE.md**](USAGE.md) - Comprehensive usage examples
- [**CLAUDE.md**](CLAUDE.md) - Architecture and development guide
- [**Examples/**](examples/) - Python examples demonstrating key features

## 🔧 Components

### Translation Pipelines
- **IndicTrans2Translator**: State-of-the-art Indic language translation
- **NLLBTranslator**: Alternative with broader language support
- **HindiPivotTranslator**: Free translation for unsupported languages
- **ClaudeTranslator**: API-based enhancement with cost tracking

### Utilities
- **Config**: Centralized configuration management
- **CostTracker**: Real-time API cost monitoring
- **QualityMetrics**: Translation quality assessment (BLEU, chrF, TER)
- **TieredOrchestrator**: Intelligent routing across translation methods

### Scripts
- `scripts/download.py` - Download datasets and models
- `scripts/translate.py` - Main translation interface
- `scripts/translate_corpus.py` - Translate entire corpus structures (NEW!)
- `scripts/verify.py` - Verify setup and downloads

## 📦 Data Sources

### Samanantar Corpus
- 49.7 million English-Indic parallel sentence pairs
- 11 Indian languages
- ~60 GB uncompressed

### Models
- **IndicTrans2**: 1B and 200M parameter models
- **NLLB-200**: 600M, 1.3B, and 3.3B models
- **LLMs**: Llama, Mistral, Aya (optional)

## 🎓 Examples

### Example 1: Basic Translation
```bash
python examples/example_1_basic_translation.py
```

### Example 2: Hindi Pivot
```bash
python examples/example_2_hindi_pivot.py
```

### Example 3: Tiered Strategy
```bash
python examples/example_3_tiered_strategy.py
```

### Example 4: Quality Assessment
```bash
python examples/example_4_quality_assessment.py
```

### Example 5: Corpus Translation (NEW!)
```bash
python examples/example_5_corpus_translation.py
```

## 🔬 Quality Assessment

Built-in metrics for evaluating translation quality:

```python
from src.quality import QualityMetrics

metrics = QualityMetrics()
scores = metrics.calculate_all_metrics(
    hypotheses=translations,
    references=reference_translations
)

print(f"BLEU: {scores['corpus_bleu']:.2f}")
print(f"chrF: {scores['chrf']:.2f}")
```

## 📊 Cost Tracking

Automatic cost tracking for all API calls:

```python
from src.utils.cost_tracker import CostTracker

tracker = CostTracker('logs/cost_tracking.json')
tracker.print_summary()
```

View costs:
```bash
python -c "from src.utils.cost_tracker import CostTracker; \
           CostTracker('logs/cost_tracking.json').print_summary()"
```

## 🛠️ Configuration

Edit `config/config.yaml` to customize:
- Language settings
- Model paths
- Translation strategies
- Tier percentages
- Cost tracking
- Processing parameters

Example custom tier distribution:
```yaml
strategy:
  unsupported_strategy:
    tiers:
      - name: "free_pivot"
        percentage: 85
      - name: "quality_enhancement"
        percentage: 10
      - name: "premium_quality"
        percentage: 5
```

## 🧪 Testing

Run verification:
```bash
# Verify full setup
python scripts/verify.py --all

# Verify specific components
python scripts/verify.py --environment
python scripts/verify.py --samanantar --languages hi bn
python scripts/verify.py --models
```

## 📈 Workflow Recommendations

1. **Start Free**: Use IndicTrans2/Hindi pivot for initial translations
2. **Assess Quality**: Evaluate on sample data (1000+ pairs)
3. **Estimate Costs**: Use cost estimation for API usage
4. **Optimize Strategy**: Adjust tier percentages based on quality/budget
5. **Monitor Spending**: Track costs in real-time
6. **Iterate**: Refine based on results and feedback

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [AI4Bharat](https://ai4bharat.org/) for IndicTrans2 and Samanantar corpus
- [Meta AI](https://ai.facebook.com/) for NLLB-200
- [Anthropic](https://www.anthropic.com/) for Claude API
- [HuggingFace](https://huggingface.co/) for datasets and models

## 📚 References

- **Samanantar**: [Ramesh et al., 2021](https://arxiv.org/abs/2104.05596)
- **IndicTrans2**: [Gala et al., 2023](https://arxiv.org/abs/2305.16307)
- **NLLB**: [NLLB Team, 2022](https://arxiv.org/abs/2207.04672)

## 📞 Support

- **Documentation**: [SETUP.md](SETUP.md), [USAGE.md](USAGE.md)
- **Issues**: [GitHub Issues](../../issues)
- **Examples**: [examples/](examples/)

## 🗺️ Roadmap

- [ ] Jupyter notebooks with tutorials
- [ ] Web interface for translations
- [ ] Support for more languages
- [ ] Integration with more APIs (OpenAI GPT, Google Translate)
- [ ] Advanced quality filters
- [ ] Parallel processing optimization
- [ ] Docker containerization
- [ ] REST API server

## ⚡ Performance

- **IndicTrans2**: ~100-500 sentences/sec (GPU)
- **NLLB**: ~50-200 sentences/sec (GPU)
- **Batch Processing**: Optimized for throughput
- **Memory Efficient**: Streaming for large datasets

## 🎯 Use Cases

- **Research**: Create parallel corpora for NLP research
- **Language Technology**: Build translation systems for low-resource languages
- **Education**: Develop language learning resources
- **Localization**: Translate content for underserved communities
- **Data Augmentation**: Generate training data for ML models

---

**Made with ❤️ for low-resource language communities**
