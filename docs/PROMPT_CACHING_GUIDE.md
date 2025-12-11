# Prompt Caching Guide for Low-Resource Indic Languages

This guide explains how to use prompt caching with Claude for cost-effective translation to low-resource Indic languages.

## Overview

Prompt caching allows you to cache large portions of your prompts (like system instructions and examples) so you pay reduced rates for cached content on subsequent requests. This is especially valuable for translation tasks where the system prompt and examples remain constant.

## Supported Languages with Caching

### Languages with Prompt Configurations

| Language | Code | Prompt File | Claude Model | Example Source |
|----------|------|-------------|--------------|----------------|
| **Awadhi** | awa | `hi_to_awa_claude.yaml` | claude-haiku-4.5 | Local corpus |
| **Braj** | bra | `hi_to_bra_claude.yaml` | claude-haiku-4.5 | Local corpus |
| **Marwari** | mwr | `hi_to_mwr_claude.yaml` | claude-haiku-4.5 | Sangrah dataset |
| **Maithili** | mai | `hi_to_mai_claude.yaml` | claude-haiku-4.5 | Sangrah dataset |
| **Nepali** | nep | `hi_to_nep_claude.yaml` | claude-haiku-4.5 | Sangrah dataset |
| **Bhojpuri** | bho | `hi_to_bho_openai.yaml` | GPT-4o-mini | Local corpus |
| **Magahi** | mag | `hi_to_mag_openai.yaml` | GPT-4o-mini | Local corpus |

## Cost Savings with Prompt Caching

### Without Caching (Standard Pricing)
- Input: ₹85 per 1M tokens
- Output: ₹425 per 1M tokens

### With Caching (Claude Haiku 4.5)
- **Cached input**: ₹8.5 per 1M tokens (90% discount!)
- Regular input: ₹85 per 1M tokens
- Output: ₹425 per 1M tokens

### Example Cost Calculation

For 10,000 Hindi sentences translated to Awadhi:

**Without caching:**
- System prompt + examples: ~5,000 tokens × 10,000 requests = 50B tokens
- User prompts: ~2,000 tokens × 10,000 = 20M tokens
- Output: ~2,000 tokens × 10,000 = 20M tokens
- **Total cost**: (50,000 + 20) × ₹85 + 20 × ₹425 = ₹4,259,700

**With caching:**
- First request: 5,000 tokens × ₹85 = ₹425
- Cached (9,999 requests): 5,000 × 9,999 × ₹8.5 = ₹424,957
- User prompts: 20M × ₹85 = ₹1,700,000
- Output: 20M × ₹425 = ₹8,500,000
- **Total cost**: ₹10,625,382 → **~₹425,957 for caching + ₹10,200,000 for generation**

**Savings on cacheable content: ~90%**

## Setting Up Examples from Sangrah Dataset

### 1. Install Dependencies

```bash
conda activate bhaashik-translation
pip install datasets huggingface-hub
```

### 2. Authenticate with HuggingFace

```bash
huggingface-cli login
# Enter your token from https://huggingface.co/settings/tokens
```

### 3. Extract Examples from Sangrah

```bash
cd /path/to/BhaashikLowsResLangTrans
python scripts/extract_examples_from_sangrah.py
```

This will:
- Download Sangrah dataset for Nepali, Maithili, and Marwari
- Extract 100 high-quality monolingual examples per language
- Filter for appropriate length and quality
- Save to `universal_translate/data/examples/`

### 4. Verify Examples

```bash
# Check Marwari examples
cat universal_translate/data/examples/hi_to_mwr_monolingual.json | jq '.num_examples'

# View first 5 examples
cat universal_translate/data/examples/hi_to_nep_monolingual.json | jq '.examples[:5]'
```

## Using Prompts with Caching

### Python Example

```python
from universal_translate.prompts import PromptManager
from anthropic import Anthropic

# Load prompt configuration
prompt_manager = PromptManager(
    config_path="universal_translate/config/prompts/hi_to_mwr_claude.yaml"
)

# Initialize Claude client
client = Anthropic(api_key="your-api-key")

# Prepare the cached system prompt with examples
system_prompt = prompt_manager.get_system_prompt()
examples_text = prompt_manager.format_examples_for_prompt(max_examples=100)

# Combine system prompt with examples
full_system_prompt = f"{system_prompt}\n\nEXAMPLES:\n{examples_text}"

# For caching, we need to structure the request properly
# The last part of the system content will be cached

hindi_text = "मैं आज बाजार जा रहा हूं।"

# Make request with caching
response = client.messages.create(
    model="claude-haiku-4.5",
    max_tokens=4096,
    system=[
        {
            "type": "text",
            "text": full_system_prompt,
            "cache_control": {"type": "ephemeral"}  # This enables caching
        }
    ],
    messages=[
        {
            "role": "user",
            "content": prompt_manager.get_user_prompt(text=hindi_text)
        }
    ]
)

print(response.content[0].text)

# Check cache usage
print(f"Cache creation tokens: {response.usage.cache_creation_input_tokens}")
print(f"Cache read tokens: {response.usage.cache_read_input_tokens}")
print(f"Regular input tokens: {response.usage.input_tokens}")
```

### Batch Processing with Caching

```python
from pathlib import Path
from tqdm import tqdm
import time

# Load Hindi sentences
hindi_sentences = Path("input/converted/Hindi/plain-text/domain.txt").read_text().splitlines()

# Prepare cached system prompt (done once)
prompt_manager = PromptManager("universal_translate/config/prompts/hi_to_awa_claude.yaml")
system_prompt = prompt_manager.get_system_prompt()
examples_text = prompt_manager.format_examples_for_prompt(max_examples=100)
full_system_prompt = f"{system_prompt}\n\nEXAMPLES:\n{examples_text}"

client = Anthropic(api_key="your-api-key")

translations = []
cache_hits = 0
total_cost = 0

for hindi_text in tqdm(hindi_sentences):
    response = client.messages.create(
        model="claude-haiku-4.5",
        max_tokens=4096,
        system=[
            {
                "type": "text",
                "text": full_system_prompt,
                "cache_control": {"type": "ephemeral"}
            }
        ],
        messages=[
            {
                "role": "user",
                "content": prompt_manager.get_user_prompt(text=hindi_text)
            }
        ]
    )

    translations.append(response.content[0].text)

    # Track cache usage
    if response.usage.cache_read_input_tokens > 0:
        cache_hits += 1

    # Calculate cost (approximate)
    cache_read_cost = response.usage.cache_read_input_tokens * 8.5 / 1_000_000
    input_cost = response.usage.input_tokens * 85 / 1_000_000
    output_cost = response.usage.output_tokens * 425 / 1_000_000
    total_cost += cache_read_cost + input_cost + output_cost

    # Rate limiting
    time.sleep(0.1)

print(f"\nCache hit rate: {cache_hits}/{len(hindi_sentences)}")
print(f"Total cost: ₹{total_cost:.2f}")
```

## Optimizing Cache Usage

### 1. Structure Your Prompts for Maximum Caching

The key is to put static content (system instructions, examples) at the end of your system messages:

```python
system=[
    {
        "type": "text",
        "text": "Basic instructions..."  # Not cached
    },
    {
        "type": "text",
        "text": "Language-specific guidelines\n\nEXAMPLES:\n...",
        "cache_control": {"type": "ephemeral"}  # This part is cached
    }
]
```

### 2. Use Enough Examples to Justify Caching

- **Minimum tokens for caching**: ~1,024 tokens
- **Recommended**: 50-100 examples (5,000-10,000 tokens)
- More examples = more cache benefit

### 3. Cache Duration

- Caches last for 5 minutes of inactivity
- Plan your batch processing to use cache before expiration
- Process related translations in batches

### 4. Monitor Cache Performance

Always check the usage statistics:

```python
usage = response.usage
print(f"Cache creation: {usage.cache_creation_input_tokens}")  # First request only
print(f"Cache hits: {usage.cache_read_input_tokens}")  # Subsequent requests
print(f"Regular input: {usage.input_tokens}")  # Non-cached input
print(f"Output: {usage.output_tokens}")
```

## Best Practices

### 1. Organize by Language Pairs

Keep separate prompt configurations for each language pair:
- `hi_to_awa_claude.yaml` for Hindi → Awadhi
- `hi_to_mwr_claude.yaml` for Hindi → Marwari
- etc.

### 2. Update Examples Periodically

Refresh examples from Sangrah dataset quarterly:

```bash
# Re-run extraction with new random seed
python scripts/extract_examples_from_sangrah.py
```

### 3. A/B Test Example Count

Test different numbers of examples:
- 25 examples: Faster, lower cache benefit
- 50 examples: Balanced
- 100 examples: Maximum quality and cache benefit

### 4. Combine with Batch API

For even more savings, combine caching with Claude's Batch API (50% off output):

```python
# Use Anthropic's Batch API
# Output cost: ₹425 → ₹212.50 per 1M tokens
# Plus 90% caching discount on system prompts
```

## Language-Specific Notes

### Awadhi (awa)
- Literary language with rich classical tradition
- Examples should include both formal and colloquial usage
- Consider including Ramcharitmanas-style examples

### Braj (bra)
- Strong devotional poetry tradition
- Include examples from Krishna bhakti literature
- Musical and rhythmic patterns important

### Marwari (mwr)
- Available in Sangrah dataset
- Commercial and trade vocabulary important
- Folk song examples useful

### Maithili (mai)
- Rich Sangrah dataset availability
- Complex honorific system
- Vidyapati-style poetic examples valuable

### Nepali (nep)
- Best coverage in Sangrah dataset
- Can be enhanced beyond IndicTrans2
- Himalayan cultural context important

## Troubleshooting

### Cache Not Working

1. Verify cache_control is set:
   ```python
   "cache_control": {"type": "ephemeral"}
   ```

2. Check minimum token threshold (~1,024 tokens)

3. Verify Claude model supports caching (Haiku 4, Sonnet 4+)

### High Costs

1. Monitor cache hit rate (should be >95% after first request)
2. Check cache expiration (5 minutes of inactivity)
3. Verify examples are being loaded correctly

### Poor Translation Quality

1. Increase number of examples (50 → 100)
2. Add domain-specific examples
3. Adjust temperature (0.1-0.5 for consistency)
4. Consider using Sonnet 4.5 for premium quality

## References

- [Claude Prompt Caching Documentation](https://docs.anthropic.com/claude/docs/prompt-caching)
- [Sangrah Dataset](https://huggingface.co/datasets/ai4bharat/sangrah)
- [AI4Bharat Project](https://ai4bharat.org/)
- [IndicTrans2 Models](https://github.com/AI4Bharat/IndicTrans2)

## Support

For issues or questions:
1. Check HuggingFace dataset documentation
2. Review Claude API documentation
3. Open an issue on GitHub with cost/cache statistics
