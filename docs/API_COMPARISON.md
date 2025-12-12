# API Comparison: OpenAI vs Claude for Low-Resource Translation

Quick guide to choosing between OpenAI and Claude APIs for translating to low-resource Indic languages.

## TL;DR

**Use OpenAI** if you:
- Can't access Claude API (payment method restrictions)
- Want good quality at lower cost
- Prefer simpler billing (credit card, more payment options)

**Use Claude** if you:
- Can access Claude API
- Want maximum cost savings (90% vs 50%)
- Need highest quality translations

## Cost Comparison

### For 10,000 Sentences (Hindi → Marwari)

| Provider | Model | Without Cache | With Cache | Savings |
|----------|-------|--------------|------------|---------|
| **OpenAI** | GPT-4o-mini | ₹701 | ₹383 | 45% |
| **Claude** | Haiku 4.5 | ₹10,625 | ₹426 | 96% |

### Per 1M Tokens (INR)

| Provider | Model | Input | Cached Input | Output |
|----------|-------|-------|--------------|--------|
| **OpenAI** | GPT-4o-mini | ₹12.75 | ₹6.38 (50% off) | ₹51 |
| **OpenAI** | GPT-4o | ₹425 | ₹212.50 (50% off) | ₹1,275 |
| **Claude** | Haiku 4.5 | ₹85 | ₹8.5 (90% off) | ₹425 |
| **Claude** | Sonnet 4.5 | ₹255 | ₹25.5 (90% off) | ₹1,275 |

## Quality Comparison

Based on informal testing with Indic languages:

| Model | Quality | Speed | Best For |
|-------|---------|-------|----------|
| GPT-4o-mini | Good | Fast | Bulk translation, cost-sensitive |
| GPT-4o | Excellent | Medium | Premium content |
| Claude Haiku 4.5 | Excellent | Very fast | Bulk translation with caching |
| Claude Sonnet 4.5 | Superior | Medium | Literary/cultural content |

## Supported Languages

Both platforms support all target languages:
- ✅ Awadhi (awa)
- ✅ Braj (bra)
- ✅ Marwari (mwr)
- ✅ Maithili (mai)
- ✅ Nepali (nep)
- ✅ Bhojpuri (bho) - existing
- ✅ Magahi (mag) - existing

## Prompt Caching Features

### OpenAI
- **Discount**: 50% on cached input
- **Minimum**: 1,024 tokens
- **TTL**: 5-10 minutes
- **How it works**: Automatic for system messages >1024 tokens

### Claude
- **Discount**: 90% on cached input
- **Minimum**: ~1,024 tokens
- **TTL**: 5 minutes
- **How it works**: Use `cache_control` parameter

## Payment Methods

### OpenAI
- ✅ Credit cards (Visa, Mastercard, Amex)
- ✅ Debit cards
- ✅ More international payment options
- ✅ Prepaid credits

### Claude
- ❌ More restrictive payment methods
- ❌ May not accept some international cards
- ❌ Limited payment options

## Example Usage

### OpenAI

```python
from openai import OpenAI
from universal_translate.prompts import PromptManager

# Load configuration
pm = PromptManager("universal_translate/config/prompts/hi_to_mwr_openai.yaml")
system_prompt = pm.get_system_prompt()
examples = pm.format_examples_for_prompt(max_examples=100)

# Translate with caching
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": f"{system_prompt}\n\nEXAMPLES:\n{examples}"},
        {"role": "user", "content": "मैं घर जा रहा हूं।"}
    ]
)
```

### Claude

```python
from anthropic import Anthropic
from universal_translate.prompts import PromptManager

# Load configuration
pm = PromptManager("universal_translate/config/prompts/hi_to_mwr_claude.yaml")
system_prompt = pm.get_system_prompt()
examples = pm.format_examples_for_prompt(max_examples=100)

# Translate with caching
client = Anthropic()
response = client.messages.create(
    model="claude-haiku-4.5",
    system=[{
        "type": "text",
        "content": f"{system_prompt}\n\nEXAMPLES:\n{examples}",
        "cache_control": {"type": "ephemeral"}
    }],
    messages=[{"role": "user", "content": "मैं घर जा रहा हूं।"}]
)
```

## Recommendations

### For Budget-Conscious Projects
**OpenAI GPT-4o-mini** - Best value for money with good quality

### For Maximum Quality
**Claude Sonnet 4.5** - Superior linguistic understanding, especially for cultural nuances

### For Bulk Translation
**OpenAI GPT-4o-mini** or **Claude Haiku 4.5** - Both excellent with caching

### For Literary/Cultural Content
**Claude Sonnet 4.5** - Better at preserving poetic/cultural elements

## Getting Started

### OpenAI Setup
```bash
# Install
pip install openai

# Set API key
export OPENAI_API_KEY="your-key"

# Run example
python examples/translate_with_openai_caching.py
```

### Claude Setup
```bash
# Install
pip install anthropic

# Set API key
export ANTHROPIC_API_KEY="your-key"

# Run example
python examples/translate_with_caching.py
```

## Documentation

- **OpenAI Guide**: [docs/OPENAI_PROMPT_CACHING_GUIDE.md](OPENAI_PROMPT_CACHING_GUIDE.md)
- **Claude Guide**: [docs/PROMPT_CACHING_GUIDE.md](PROMPT_CACHING_GUIDE.md)

## Troubleshooting

### OpenAI Issues
- Check API key: `echo $OPENAI_API_KEY`
- Verify billing: https://platform.openai.com/account/billing
- Test: `openai api chat_completions.create -m gpt-4o-mini -g user "test"`

### Claude Issues
- Check API key: `echo $ANTHROPIC_API_KEY`
- Verify access: https://console.anthropic.com/
- Payment methods: May need specific card types

## Cost Optimization Tips

1. **Use prompt caching** - Set up >1024 token system prompts
2. **Batch processing** - Process multiple sentences in one session
3. **Monitor usage** - Track cache hit rates
4. **Choose right model** - Use mini/haiku for bulk, premium models for quality
5. **Extract good examples** - 50-100 examples from Sangrah dataset
6. **Process within TTL** - Complete batches before cache expires

## Summary

| Factor | OpenAI Winner | Claude Winner |
|--------|--------------|--------------|
| Accessibility | ✅ | |
| Cost (absolute) | ✅ | |
| Cost (with caching) | | ✅ |
| Quality (general) | | ✅ |
| Speed | | ✅ |
| Payment options | ✅ | |
| Cache discount | | ✅ |

**Bottom line**: Use OpenAI for accessibility and good value. Use Claude for maximum savings and quality if you can access it.
