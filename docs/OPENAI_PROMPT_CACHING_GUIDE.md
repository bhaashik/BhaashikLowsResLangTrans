# OpenAI Prompt Caching Guide for Low-Resource Indic Languages

This guide explains how to use OpenAI's prompt caching with GPT-4o-mini for cost-effective translation to low-resource Indic languages.

## Overview

OpenAI's prompt caching allows you to cache large portions of your prompts (like system instructions and examples) with a **50% discount** on cached input tokens. While not as dramatic as Claude's 90% savings, it still provides significant cost reduction for translation workflows.

## Supported Languages with Caching

### Languages with OpenAI Prompt Configurations

| Language | Code | Prompt File | Model | Example Source |
|----------|------|-------------|-------|----------------|
| **Awadhi** | awa | `hi_to_awa_openai.yaml` | gpt-4o-mini | Local corpus |
| **Braj** | bra | `hi_to_bra_openai.yaml` | gpt-4o-mini | Local corpus |
| **Marwari** | mwr | `hi_to_mwr_openai.yaml` | gpt-4o-mini | Sangrah dataset |
| **Maithili** | mai | `hi_to_mai_openai.yaml` | gpt-4o-mini | Sangrah dataset |
| **Nepali** | nep | `hi_to_nep_openai.yaml` | gpt-4o-mini | Sangrah dataset |
| **Bhojpuri** | bho | `hi_to_bho_openai.yaml` | gpt-4o-mini | Local corpus |
| **Magahi** | mag | `hi_to_mag_openai.yaml` | gpt-4o-mini | Local corpus |

## Cost Savings with OpenAI Prompt Caching

### GPT-4o-mini Pricing (USD)
- **Standard input**: $0.15 per 1M tokens
- **Cached input**: $0.075 per 1M tokens (50% discount!)
- **Output**: $0.60 per 1M tokens

### INR Pricing (at ₹85/$)
- **Standard input**: ₹12.75 per 1M tokens
- **Cached input**: ₹6.38 per 1M tokens (50% discount!)
- **Output**: ₹51 per 1M tokens

### Caching Requirements
- Minimum tokens for caching: **1,024 tokens**
- Cache TTL: **5-10 minutes** of inactivity
- Eligible for caching: System messages and beginning of user messages

### Example Cost Calculation

For 10,000 Hindi sentences translated to Marwari:

**Without caching:**
- System prompt + examples: ~5,000 tokens × 10,000 requests = 50M tokens
- User prompts: ~100 tokens × 10,000 = 1M tokens
- Output: ~100 tokens × 10,000 = 1M tokens
- **Total cost**: (50 + 1) × ₹12.75 + 1 × ₹51 = ₹650.25 + ₹51 = **₹701.25**

**With caching:**
- First request (cache creation): 5,000 tokens × ₹12.75/M = ₹0.06
- Cached (9,999 requests): 5,000 × 9,999 × ₹6.38/M = ₹319.17
- User prompts: 1M × ₹12.75 = ₹12.75
- Output: 1M × ₹51 = ₹51
- **Total cost**: ₹0.06 + ₹319.17 + ₹12.75 + ₹51 = **₹382.98**

**Savings**: ₹318.27 (45% reduction) on total cost, 50% on cached tokens

## Setting Up Examples from Sangrah Dataset

### 1. Install Dependencies

```bash
conda activate bhaashik-translation
pip install openai datasets huggingface-hub
```

### 2. Extract Examples

```bash
# Authenticate with HuggingFace
huggingface-cli login

# Extract examples from Sangrah
python scripts/extract_examples_from_sangrah.py
```

This extracts 100 high-quality monolingual examples for:
- Nepali (from Sangrah)
- Maithili (from Sangrah)
- Marwari (from Sangrah)

## Using Prompts with OpenAI Caching

### Python Example - Single Translation

```python
from universal_translate.prompts import PromptManager
from openai import OpenAI
import os

# Load prompt configuration
prompt_manager = PromptManager(
    config_path="universal_translate/config/prompts/hi_to_mwr_openai.yaml"
)

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Prepare system prompt with examples
system_prompt = prompt_manager.get_system_prompt()
examples_text = prompt_manager.format_examples_for_prompt(max_examples=100)
full_system_prompt = f"{system_prompt}\n\nEXAMPLES:\n{examples_text}"

# Hindi text to translate
hindi_text = "मैं आज बाजार जा रहा हूं।"

# Make request with caching
# OpenAI automatically caches system messages >1024 tokens
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "system",
            "content": full_system_prompt  # Automatically cached if >1024 tokens
        },
        {
            "role": "user",
            "content": prompt_manager.get_user_prompt(text=hindi_text)
        }
    ],
    temperature=0.3,
    max_tokens=2048,
    top_p=0.9
)

print(response.choices[0].message.content)

# Check cache usage (available in response)
print(f"Prompt tokens: {response.usage.prompt_tokens}")
print(f"Cached tokens: {response.usage.prompt_tokens_details.cached_tokens if hasattr(response.usage, 'prompt_tokens_details') else 0}")
print(f"Completion tokens: {response.usage.completion_tokens}")
```

### Batch Processing with Caching

```python
from pathlib import Path
from tqdm import tqdm
import time

# Load Hindi sentences
hindi_sentences = Path("input/converted/Hindi/plain-text/domain.txt").read_text().splitlines()

# Prepare system prompt (done once)
prompt_manager = PromptManager("universal_translate/config/prompts/hi_to_awa_openai.yaml")
system_prompt = prompt_manager.get_system_prompt()
examples_text = prompt_manager.format_examples_for_prompt(max_examples=100)
full_system_prompt = f"{system_prompt}\n\nEXAMPLES:\n{examples_text}"

client = OpenAI()

translations = []
cache_hits = 0
total_cost_inr = 0

for hindi_text in tqdm(hindi_sentences):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": full_system_prompt  # Cached after first request
            },
            {
                "role": "user",
                "content": prompt_manager.get_user_prompt(text=hindi_text)
            }
        ],
        temperature=0.3,
        max_tokens=2048
    )

    translations.append(response.choices[0].message.content)

    # Track cache usage
    usage = response.usage
    cached_tokens = usage.prompt_tokens_details.cached_tokens if hasattr(usage, 'prompt_tokens_details') else 0

    if cached_tokens > 0:
        cache_hits += 1

    # Calculate cost (INR)
    uncached_tokens = usage.prompt_tokens - cached_tokens
    cached_cost = cached_tokens * 6.38 / 1_000_000  # 50% discount
    uncached_cost = uncached_tokens * 12.75 / 1_000_000
    output_cost = usage.completion_tokens * 51 / 1_000_000
    total_cost_inr += cached_cost + uncached_cost + output_cost

    # Rate limiting
    time.sleep(0.05)  # Avoid rate limits

print(f"\nCache hit rate: {cache_hits}/{len(hindi_sentences)}")
print(f"Total cost: ₹{total_cost_inr:.2f}")
print(f"Average per sentence: ₹{total_cost_inr/len(hindi_sentences):.4f}")
```

## Optimizing Cache Usage

### 1. Ensure System Prompt is >1024 Tokens

OpenAI requires at least 1,024 tokens for caching:

```python
# Check system prompt token count
import tiktoken

encoding = tiktoken.encoding_for_model("gpt-4o-mini")
tokens = encoding.encode(full_system_prompt)
print(f"System prompt tokens: {len(tokens)}")

# If <1024, add more examples
if len(tokens) < 1024:
    examples_text = prompt_manager.format_examples_for_prompt(max_examples=150)
    full_system_prompt = f"{system_prompt}\n\nEXAMPLES:\n{examples_text}"
```

### 2. Process in Batches Within Cache TTL

Cache expires after 5-10 minutes of inactivity:

```python
from datetime import datetime, timedelta

# Process all requests within 5 minutes
batch_size = 100
start_time = datetime.now()

for i in range(0, len(hindi_sentences), batch_size):
    batch = hindi_sentences[i:i+batch_size]

    for hindi_text in batch:
        # Translate...
        pass

    # Check if approaching cache expiration
    elapsed = datetime.now() - start_time
    if elapsed > timedelta(minutes=4):
        print("Cache may expire soon, processed batch quickly")
        start_time = datetime.now()  # Reset timer
```

### 3. Monitor Cache Performance

```python
# Track cache statistics
total_prompt_tokens = 0
total_cached_tokens = 0

for response in responses:
    usage = response.usage
    total_prompt_tokens += usage.prompt_tokens

    if hasattr(usage, 'prompt_tokens_details'):
        total_cached_tokens += usage.prompt_tokens_details.cached_tokens

cache_efficiency = total_cached_tokens / total_prompt_tokens if total_prompt_tokens > 0 else 0
print(f"Cache efficiency: {cache_efficiency:.1%}")
print(f"Savings: ₹{total_cached_tokens * 6.38 / 1_000_000:.2f} on cached tokens")
```

## Best Practices

### 1. Use Enough Examples to Exceed 1024 Tokens

```python
# Aim for 1500-2000 tokens in system prompt
# ~50-100 examples usually achieves this
examples = prompt_manager.get_examples(max_examples=75)
```

### 2. Keep System Prompt Static

Don't vary the system prompt between requests:

```python
# GOOD - Same system prompt cached
for hindi_text in sentences:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": static_system_prompt},  # Cached
            {"role": "user", "content": hindi_text}
        ]
    )

# BAD - Different system prompts, no caching benefit
for hindi_text in sentences:
    system_prompt = f"Translate to Marwari. Today is {datetime.now()}"  # Changes!
    # No caching
```

### 3. Use gpt-4o-mini for Cost Efficiency

```python
# Most cost-effective model with caching
model = "gpt-4o-mini"  # ₹12.75/M input, ₹51/M output

# For higher quality, use gpt-4o
model = "gpt-4o"  # ₹425/M input, ₹1,275/M output
# But caching still saves 50% on input
```

### 4. Combine with Batch API for More Savings

OpenAI's Batch API provides 50% discount on output:

```python
# Create batch file with all translations
batch_requests = [
    {
        "custom_id": f"request-{i}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": hindi_text}
            ]
        }
    }
    for i, hindi_text in enumerate(hindi_sentences)
]

# Submit batch
# Output cost: ₹51 → ₹25.50 per 1M tokens
```

## Language-Specific Notes

### Awadhi (awa)
- Literary examples from Ramcharitmanas useful
- Include both formal and colloquial usage

### Braj (bra)
- Krishna bhakti poetry examples valuable
- Musical and rhythmic patterns important

### Marwari (mwr)
- Use Sangrah dataset examples
- Commercial vocabulary important
- Folk expressions enhance quality

### Maithili (mai)
- Excellent Sangrah dataset coverage
- Complex honorific system needs many examples
- Vidyapati-style poetic examples valuable

### Nepali (nep)
- Best Sangrah coverage
- Can enhance IndicTrans2 output
- Himalayan cultural context important

## Troubleshooting

### Cache Not Working

1. **Check token count**:
   ```python
   tokens = tiktoken.encoding_for_model("gpt-4o-mini").encode(system_prompt)
   print(f"Tokens: {len(tokens)} (need >1024)")
   ```

2. **Verify cache in response**:
   ```python
   if hasattr(response.usage, 'prompt_tokens_details'):
       print(f"Cached: {response.usage.prompt_tokens_details.cached_tokens}")
   else:
       print("No cache details in response")
   ```

3. **Check cache expiration** (5-10 minutes):
   ```python
   import time
   # Process quickly to maintain cache
   time.sleep(0.1)  # Minimal delay between requests
   ```

### High Costs

1. Monitor cache hit rate (should be >95% after first request)
2. Verify examples are being loaded
3. Check that system prompt is consistent

### Poor Translation Quality

1. Increase examples (50 → 100)
2. Add domain-specific examples
3. Lower temperature (0.3 → 0.1 for consistency)
4. Consider gpt-4o for premium quality

## Cost Comparison

| Model | Input (₹/1M) | Cached (₹/1M) | Output (₹/1M) | Best For |
|-------|-------------|---------------|---------------|----------|
| gpt-4o-mini | 12.75 | 6.38 (50% off) | 51 | Bulk translation, cost-sensitive |
| gpt-4o | 425 | 212.50 (50% off) | 1,275 | Premium quality, important content |
| gpt-3.5-turbo | 42.50 | No caching | 127.50 | Legacy (use gpt-4o-mini instead) |

## References

- [OpenAI Prompt Caching Documentation](https://platform.openai.com/docs/guides/prompt-caching)
- [Sangrah Dataset](https://huggingface.co/datasets/ai4bharat/sangrah)
- [AI4Bharat Project](https://ai4bharat.org/)
- [GPT-4o-mini Model Card](https://platform.openai.com/docs/models/gpt-4o-mini)

## Example Usage

See `examples/translate_with_openai_caching.py` for a complete working example with cost tracking and cache monitoring.
