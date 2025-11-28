#!/usr/bin/env python3
"""
Basic usage examples for universal_translate package.

This script demonstrates:
1. Loading prompt configurations
2. Creating translation providers
3. Translating with custom prompts
4. Cost estimation
5. Using translation examples
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from universal_translate import (
    AnthropicProvider,
    PromptManager,
    TranslationRequest,
    TranslationUnit,
    TranslationStatus
)


def example_1_basic_translation():
    """Example 1: Basic translation without prompts."""
    print("="*70)
    print("EXAMPLE 1: Basic Translation (No Custom Prompt)")
    print("="*70)

    # Create provider without prompt manager (uses default prompting)
    provider = AnthropicProvider(model="claude-haiku-4.5")

    # Create translation request
    request = TranslationRequest(
        units=[
            TranslationUnit(text="Hello, how are you?", index=0),
            TranslationUnit(text="The weather is nice today.", index=1)
        ],
        src_lang="en",
        tgt_lang="hi"
    )

    # Translate
    with provider:
        response = provider.translate_sync(request)

    # Print results
    for result in response.results:
        if result.status == TranslationStatus.COMPLETED:
            print(f"\nSource: {result.source}")
            print(f"Translation: {result.translation}")
            print(f"Cost: ₹{result.cost:.4f}")

    print(f"\nTotal Cost: ₹{response.total_cost:.2f}")
    print()


def example_2_with_custom_prompt():
    """Example 2: Translation with custom prompt configuration."""
    print("="*70)
    print("EXAMPLE 2: Translation with Custom Prompt")
    print("="*70)

    # Load prompt configuration
    prompt_config = Path(__file__).parent.parent / "config/prompts/low_resource.yaml"
    prompt_mgr = PromptManager(str(prompt_config))

    print(f"\nLoaded prompt: {prompt_mgr.config.get('name')}")
    print(f"Description: {prompt_mgr.config.get('description')}")
    print(f"Caching enabled: {prompt_mgr.supports_caching()}")
    print(f"Number of examples: {len(prompt_mgr.examples)}")

    # Create provider with prompt
    provider = AnthropicProvider(
        model="claude-haiku-4.5",
        prompt_manager=prompt_mgr
    )

    # Create request
    request = TranslationRequest(
        units=[
            TranslationUnit(text="राम घर जा रहे हैं।", index=0),
            TranslationUnit(text="यह किताब अच्छी है।", index=1)
        ],
        src_lang="hi",
        tgt_lang="bho"
    )

    # Translate
    with provider:
        response = provider.translate_sync(request)

    # Print results
    for result in response.results:
        if result.status == TranslationStatus.COMPLETED:
            print(f"\nSource: {result.source}")
            print(f"Translation: {result.translation}")
            print(f"Cost: ₹{result.cost:.4f}")
            print(f"Metadata: {result.metadata}")

    print(f"\nTotal Cost: ₹{response.total_cost:.2f}")
    print()


def example_3_cost_estimation():
    """Example 3: Cost estimation before translation."""
    print("="*70)
    print("EXAMPLE 3: Cost Estimation")
    print("="*70)

    provider = AnthropicProvider(model="claude-haiku-4.5")

    # Create a large batch request
    texts = [
        "यह एक परीक्षण वाक्य है।",
        "मुझे अनुवाद करना पसंद है।",
        "यह बहुत रोचक है।"
    ] * 100  # 300 sentences

    request = TranslationRequest(
        units=[TranslationUnit(text=text, index=i) for i, text in enumerate(texts)],
        src_lang="hi",
        tgt_lang="bho"
    )

    # Get cost estimate
    estimate = provider.get_cost_estimate(request)

    print(f"\nTranslation units: {estimate.num_units}")
    print(f"Estimated tokens: {estimate.estimated_tokens:,}")
    print(f"Input cost: ₹{estimate.input_cost:.2f}")
    print(f"Output cost: ₹{estimate.output_cost:.2f}")
    print(f"Total estimated cost: ₹{estimate.total_cost:.2f}")
    print(f"Cost per unit: ₹{estimate.total_cost/estimate.num_units:.4f}")

    # Budget check
    budget = 50  # ₹50 budget
    if estimate.total_cost < budget:
        print(f"\n✓ Within budget (₹{budget}). Proceeding with translation...")
    else:
        print(f"\n✗ Exceeds budget (₹{budget}). Consider using batch API or different model.")

    print()


def example_4_batch_api():
    """Example 4: Using Batch API for cost savings."""
    print("="*70)
    print("EXAMPLE 4: Batch API for Cost Savings")
    print("="*70)

    # Regular API
    provider_regular = AnthropicProvider(
        model="claude-haiku-4.5",
        use_batch_api=False
    )

    # Batch API (50% off output)
    provider_batch = AnthropicProvider(
        model="claude-haiku-4.5",
        use_batch_api=True
    )

    request = TranslationRequest(
        units=[TranslationUnit(text="परीक्षण वाक्य।", index=0)] * 10,
        src_lang="hi",
        tgt_lang="bho"
    )

    # Compare estimates
    estimate_regular = provider_regular.get_cost_estimate(request)
    estimate_batch = provider_batch.get_cost_estimate(request)

    print(f"\nRegular API cost: ₹{estimate_regular.total_cost:.2f}")
    print(f"Batch API cost: ₹{estimate_batch.total_cost:.2f}")
    print(f"Savings: ₹{estimate_regular.total_cost - estimate_batch.total_cost:.2f} ({((estimate_regular.total_cost - estimate_batch.total_cost)/estimate_regular.total_cost)*100:.1f}%)")

    print()


def example_5_prompt_inspection():
    """Example 5: Inspecting rendered prompts."""
    print("="*70)
    print("EXAMPLE 5: Prompt Inspection")
    print("="*70)

    # Load prompt
    prompt_config = Path(__file__).parent.parent / "config/prompts/general.yaml"
    prompt_mgr = PromptManager(str(prompt_config))

    # Render system prompt
    system = prompt_mgr.get_system_prompt(
        source_lang="Hindi",
        target_lang="Bhojpuri"
    )

    # Render user prompt
    user = prompt_mgr.get_user_prompt(
        text="यह एक परीक्षण है।",
        source_lang="Hindi",
        target_lang="Bhojpuri"
    )

    print("\nSystem Prompt:")
    print("-" * 70)
    print(system)

    print("\nUser Prompt:")
    print("-" * 70)
    print(user)

    print()


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("UNIVERSAL TRANSLATE - USAGE EXAMPLES")
    print("="*70 + "\n")

    # Note: These examples require ANTHROPIC_API_KEY environment variable
    # Uncomment to run with actual API calls

    # example_1_basic_translation()
    # example_2_with_custom_prompt()
    example_3_cost_estimation()
    example_4_batch_api()
    example_5_prompt_inspection()

    print("="*70)
    print("All examples completed!")
    print("="*70)


if __name__ == '__main__':
    main()
