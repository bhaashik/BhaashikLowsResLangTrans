#!/usr/bin/env python3
"""
Example: Translating Hindi to low-resource languages using OpenAI with prompt caching.

This demonstrates 50% cost savings through prompt caching with GPT-4o-mini.
"""

import os
from pathlib import Path
from openai import OpenAI
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from universal_translate.prompts import PromptManager


def translate_with_openai_caching(
    hindi_text: str,
    target_language: str = "mwr",  # Marwari
    api_key: str = None
):
    """
    Translate Hindi text to target language using OpenAI with prompt caching.

    Args:
        hindi_text: Hindi text to translate
        target_language: Target language code (awa, bra, mwr, mai, nep)
        api_key: OpenAI API key (or set OPENAI_API_KEY env var)

    Returns:
        Translated text and usage statistics
    """
    # Language configuration
    LANGUAGES = {
        'awa': 'Awadhi',
        'bra': 'Braj',
        'mwr': 'Marwari',
        'mai': 'Maithili',
        'nep': 'Nepali'
    }

    if target_language not in LANGUAGES:
        raise ValueError(f"Unsupported language: {target_language}")

    # Load prompt configuration
    prompt_config = f"universal_translate/config/prompts/hi_to_{target_language}_openai.yaml"
    prompt_manager = PromptManager(config_path=prompt_config)

    # Get system prompt with examples
    system_prompt = prompt_manager.get_system_prompt()
    examples_text = prompt_manager.format_examples_for_prompt(max_examples=100)

    # Combine system prompt with examples
    full_system_prompt = f"{system_prompt}\n\nEXAMPLES:\n{examples_text}"

    # Initialize OpenAI client
    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    # Make request with caching
    # OpenAI automatically caches system messages >1024 tokens
    print(f"Translating to {LANGUAGES[target_language]}...")
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

    # Extract translation
    translation = response.choices[0].message.content

    # Get usage statistics
    usage = response.usage
    prompt_tokens = usage.prompt_tokens
    completion_tokens = usage.completion_tokens

    # Check for cached tokens (if available in response)
    cached_tokens = 0
    if hasattr(usage, 'prompt_tokens_details') and usage.prompt_tokens_details:
        cached_tokens = usage.prompt_tokens_details.cached_tokens

    uncached_tokens = prompt_tokens - cached_tokens

    # Calculate costs (INR at ₹85/$)
    # GPT-4o-mini: $0.15/1M input, $0.075/1M cached, $0.60/1M output
    uncached_cost_inr = uncached_tokens * 12.75 / 1_000_000  # ₹12.75/M
    cached_cost_inr = cached_tokens * 6.38 / 1_000_000  # ₹6.38/M (50% off)
    output_cost_inr = completion_tokens * 51 / 1_000_000  # ₹51/M
    total_cost_inr = uncached_cost_inr + cached_cost_inr + output_cost_inr

    # Print statistics
    print("\n" + "=" * 60)
    print(f"Hindi: {hindi_text}")
    print(f"{LANGUAGES[target_language]}: {translation}")
    print("=" * 60)
    print("\nToken Usage:")
    print(f"  Prompt tokens: {prompt_tokens:,}")
    print(f"    - Cached: {cached_tokens:,} tokens (50% discount)")
    print(f"    - Uncached: {uncached_tokens:,} tokens")
    print(f"  Completion tokens: {completion_tokens:,}")
    print("\nCost Breakdown (INR):")
    print(f"  Cached input (50% off): ₹{cached_cost_inr:.6f}")
    print(f"  Uncached input: ₹{uncached_cost_inr:.6f}")
    print(f"  Output: ₹{output_cost_inr:.6f}")
    print(f"  Total: ₹{total_cost_inr:.6f}")
    if cached_tokens > 0:
        savings = (cached_tokens * 12.75 / 1_000_000) - cached_cost_inr
        print(f"  Savings from caching: ₹{savings:.6f}")
    print("=" * 60)

    return translation, {
        'usage': usage,
        'cost_inr': total_cost_inr,
        'cached_tokens': cached_tokens
    }


def batch_translate_with_openai_caching(
    hindi_sentences: list,
    target_language: str = "mwr",
    api_key: str = None
):
    """
    Batch translate multiple Hindi sentences with OpenAI caching.

    First request creates cache, subsequent requests benefit from 50% discount.

    Args:
        hindi_sentences: List of Hindi sentences
        target_language: Target language code
        api_key: OpenAI API key

    Returns:
        List of translations and statistics
    """
    LANGUAGES = {
        'awa': 'Awadhi',
        'bra': 'Braj',
        'mwr': 'Marwari',
        'mai': 'Maithili',
        'nep': 'Nepali'
    }

    # Load prompt configuration
    prompt_config = f"universal_translate/config/prompts/hi_to_{target_language}_openai.yaml"
    prompt_manager = PromptManager(config_path=prompt_config)

    # Prepare system prompt
    system_prompt = prompt_manager.get_system_prompt()
    examples_text = prompt_manager.format_examples_for_prompt(max_examples=100)
    full_system_prompt = f"{system_prompt}\n\nEXAMPLES:\n{examples_text}"

    # Initialize client
    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    translations = []
    total_cost_inr = 0
    cache_hits = 0
    total_cached_tokens = 0

    print(f"Batch translating {len(hindi_sentences)} sentences to {LANGUAGES[target_language]}...")
    print("=" * 60)

    for i, hindi_text in enumerate(hindi_sentences, 1):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": full_system_prompt
                },
                {
                    "role": "user",
                    "content": prompt_manager.get_user_prompt(text=hindi_text)
                }
            ],
            temperature=0.3,
            max_tokens=2048
        )

        translation = response.choices[0].message.content
        translations.append(translation)

        # Track cache usage
        usage = response.usage
        cached_tokens = 0
        if hasattr(usage, 'prompt_tokens_details') and usage.prompt_tokens_details:
            cached_tokens = usage.prompt_tokens_details.cached_tokens

        if cached_tokens > 0:
            cache_hits += 1
            total_cached_tokens += cached_tokens

        # Calculate cost
        uncached_tokens = usage.prompt_tokens - cached_tokens
        uncached_cost = uncached_tokens * 12.75 / 1_000_000
        cached_cost = cached_tokens * 6.38 / 1_000_000
        output_cost = usage.completion_tokens * 51 / 1_000_000
        total_cost_inr += uncached_cost + cached_cost + output_cost

        # Print progress
        status = f"CACHED ({cached_tokens})" if cached_tokens > 0 else "FIRST"
        print(f"{i}/{len(hindi_sentences)} [{status}]: {hindi_text[:40]}...")

    # Calculate savings
    full_price_cached = total_cached_tokens * 12.75 / 1_000_000
    cached_price = total_cached_tokens * 6.38 / 1_000_000
    savings = full_price_cached - cached_price

    print("=" * 60)
    print(f"Cache hit rate: {cache_hits}/{len(hindi_sentences)}")
    print(f"Total cached tokens: {total_cached_tokens:,}")
    print(f"Total cost: ₹{total_cost_inr:.4f}")
    print(f"Savings from caching: ₹{savings:.4f} (50% on {total_cached_tokens:,} tokens)")
    print(f"Average cost per sentence: ₹{total_cost_inr/len(hindi_sentences):.6f}")
    print("=" * 60)

    return translations, {
        'total_cost_inr': total_cost_inr,
        'cache_hit_rate': cache_hits / len(hindi_sentences),
        'savings_inr': savings
    }


if __name__ == "__main__":
    # Example 1: Single translation to Marwari
    print("\nExample 1: Single translation to Marwari with GPT-4o-mini")
    print("=" * 60)
    hindi_text = "मैं आज बाजार जा रहा हूं और कुछ सब्जियां खरीदूंगा।"
    translation, stats = translate_with_openai_caching(
        hindi_text=hindi_text,
        target_language="mwr"
    )

    # Example 2: Batch translation to Nepali
    print("\n\nExample 2: Batch translation to Nepali with caching")
    print("=" * 60)
    hindi_sentences = [
        "आज मौसम बहुत अच्छा है।",
        "मुझे किताबें पढ़ना पसंद है।",
        "वह स्कूल जा रहा है।",
        "हमें खाना खाना चाहिए।",
        "वे घर पहुंच गए हैं।",
        "यह फूल बहुत सुंदर है।",
        "मैं कल वहां जाऊंगा।",
        "उसने मुझे एक पत्र लिखा।"
    ]
    translations, stats = batch_translate_with_openai_caching(
        hindi_sentences=hindi_sentences,
        target_language="nep"
    )

    print("\nTranslations:")
    for hindi, nepali in zip(hindi_sentences, translations):
        print(f"  HI: {hindi}")
        print(f"  NE: {nepali}")
        print()

    print(f"\nOpenAI Prompt Caching Summary:")
    print(f"  - Cache hit rate: {stats['cache_hit_rate']:.1%}")
    print(f"  - Total savings: ₹{stats['savings_inr']:.4f} (50% on cached tokens)")
    print(f"  - Cost per sentence: ₹{stats['total_cost_inr']/len(hindi_sentences):.6f}")
    print(f"\nNote: Cache benefits appear after the first request!")
