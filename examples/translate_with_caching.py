#!/usr/bin/env python3
"""
Example: Translating Hindi to low-resource languages using Claude with prompt caching.

This demonstrates 90% cost savings through prompt caching with monolingual examples.
"""

import os
from pathlib import Path
from anthropic import Anthropic
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from universal_translate.prompts import PromptManager


def translate_with_caching(
    hindi_text: str,
    target_language: str = "mwr",  # Marwari
    api_key: str = None
):
    """
    Translate Hindi text to target language using Claude with prompt caching.

    Args:
        hindi_text: Hindi text to translate
        target_language: Target language code (awa, bra, mwr, mai, nep)
        api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)

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
    prompt_config = f"universal_translate/config/prompts/hi_to_{target_language}_claude.yaml"
    prompt_manager = PromptManager(config_path=prompt_config)

    # Get system prompt with examples
    system_prompt = prompt_manager.get_system_prompt()
    examples_text = prompt_manager.format_examples_for_prompt(max_examples=100)

    # Combine system prompt with examples
    full_system_prompt = f"{system_prompt}\n\nEXAMPLES:\n{examples_text}"

    # Initialize Claude client
    client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    # Make request with caching
    print(f"Translating to {LANGUAGES[target_language]}...")
    response = client.messages.create(
        model="claude-haiku-4.5",
        max_tokens=4096,
        system=[
            {
                "type": "text",
                "text": full_system_prompt,
                "cache_control": {"type": "ephemeral"}  # Enable caching
            }
        ],
        messages=[
            {
                "role": "user",
                "content": prompt_manager.get_user_prompt(text=hindi_text)
            }
        ]
    )

    # Extract translation
    translation = response.content[0].text

    # Calculate costs (INR)
    usage = response.usage
    cache_creation_cost = usage.cache_creation_input_tokens * 85 / 1_000_000
    cache_read_cost = usage.cache_read_input_tokens * 8.5 / 1_000_000  # 90% discount
    input_cost = usage.input_tokens * 85 / 1_000_000
    output_cost = usage.output_tokens * 425 / 1_000_000
    total_cost = cache_creation_cost + cache_read_cost + input_cost + output_cost

    # Print statistics
    print("\n" + "=" * 60)
    print(f"Hindi: {hindi_text}")
    print(f"{LANGUAGES[target_language]}: {translation}")
    print("=" * 60)
    print("\nToken Usage:")
    print(f"  Cache creation: {usage.cache_creation_input_tokens:,} tokens")
    print(f"  Cache read: {usage.cache_read_input_tokens:,} tokens")
    print(f"  Regular input: {usage.input_tokens:,} tokens")
    print(f"  Output: {usage.output_tokens:,} tokens")
    print("\nCost Breakdown (INR):")
    print(f"  Cache creation: ₹{cache_creation_cost:.4f}")
    print(f"  Cache read (90% off): ₹{cache_read_cost:.4f}")
    print(f"  Regular input: ₹{input_cost:.4f}")
    print(f"  Output: ₹{output_cost:.4f}")
    print(f"  Total: ₹{total_cost:.4f}")
    print("=" * 60)

    return translation, {
        'usage': usage,
        'cost_inr': total_cost
    }


def batch_translate_with_caching(
    hindi_sentences: list,
    target_language: str = "mwr",
    api_key: str = None
):
    """
    Batch translate multiple Hindi sentences with caching.

    First request creates cache, subsequent requests use cached system prompt.

    Args:
        hindi_sentences: List of Hindi sentences
        target_language: Target language code
        api_key: Anthropic API key

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
    prompt_config = f"universal_translate/config/prompts/hi_to_{target_language}_claude.yaml"
    prompt_manager = PromptManager(config_path=prompt_config)

    # Prepare cached system prompt
    system_prompt = prompt_manager.get_system_prompt()
    examples_text = prompt_manager.format_examples_for_prompt(max_examples=100)
    full_system_prompt = f"{system_prompt}\n\nEXAMPLES:\n{examples_text}"

    # Initialize client
    client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    translations = []
    total_cost = 0
    cache_hits = 0

    print(f"Batch translating {len(hindi_sentences)} sentences to {LANGUAGES[target_language]}...")
    print("=" * 60)

    for i, hindi_text in enumerate(hindi_sentences, 1):
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

        translation = response.content[0].text
        translations.append(translation)

        # Track cache usage
        usage = response.usage
        if usage.cache_read_input_tokens > 0:
            cache_hits += 1

        # Calculate cost
        cache_creation_cost = usage.cache_creation_input_tokens * 85 / 1_000_000
        cache_read_cost = usage.cache_read_input_tokens * 8.5 / 1_000_000
        input_cost = usage.input_tokens * 85 / 1_000_000
        output_cost = usage.output_tokens * 425 / 1_000_000
        total_cost += cache_creation_cost + cache_read_cost + input_cost + output_cost

        # Print progress
        status = "CACHED" if usage.cache_read_input_tokens > 0 else "FIRST"
        print(f"{i}/{len(hindi_sentences)} [{status}]: {hindi_text[:40]}...")

    print("=" * 60)
    print(f"Cache hit rate: {cache_hits}/{len(hindi_sentences)}")
    print(f"Total cost: ₹{total_cost:.4f}")
    print(f"Average cost per sentence: ₹{total_cost/len(hindi_sentences):.4f}")
    print("=" * 60)

    return translations, {
        'total_cost_inr': total_cost,
        'cache_hit_rate': cache_hits / len(hindi_sentences)
    }


if __name__ == "__main__":
    # Example 1: Single translation to Marwari
    print("\nExample 1: Single translation to Marwari")
    print("=" * 60)
    hindi_text = "मैं आज बाजार जा रहा हूं और कुछ सब्जियां खरीदूंगा।"
    translation, stats = translate_with_caching(
        hindi_text=hindi_text,
        target_language="mwr"
    )

    # Example 2: Batch translation to Nepali
    print("\n\nExample 2: Batch translation to Nepali")
    print("=" * 60)
    hindi_sentences = [
        "आज मौसम बहुत अच्छा है।",
        "मुझे किताबें पढ़ना पसंद है।",
        "वह स्कूल जा रहा है।",
        "हमें खाना खाना चाहिए।",
        "वे घर पहुंच गए हैं।"
    ]
    translations, stats = batch_translate_with_caching(
        hindi_sentences=hindi_sentences,
        target_language="nep"
    )

    print("\nTranslations:")
    for hindi, nepali in zip(hindi_sentences, translations):
        print(f"  HI: {hindi}")
        print(f"  NE: {nepali}")
        print()

    print(f"\nCache savings: ~90% on system prompt after first request!")
    print(f"With {len(hindi_sentences)} sentences, cache hit rate: {stats['cache_hit_rate']:.1%}")
