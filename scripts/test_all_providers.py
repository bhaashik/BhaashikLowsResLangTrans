#!/usr/bin/env python3
"""
Test script to demonstrate translation with all available providers.

Usage:
    # Test all available providers
    python scripts/test_all_providers.py

    # Test specific provider
    python scripts/test_all_providers.py --provider openai

    # Test with custom text
    python scripts/test_all_providers.py --text "आपका स्वागत है।"
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from universal_translate.providers.registry import ProviderRegistry, create_translator
from universal_translate.core import (
    TranslationUnit,
    TranslationRequest,
    TranslationStatus
)


# Test sentences in Hindi
TEST_SENTENCES = [
    "नमस्ते, आप कैसे हैं?",
    "मैं अपने परिवार के साथ बाजार जा रहा हूं।",
    "यह पुस्तक बहुत रोचक है।"
]


def test_provider(
    provider_name: str,
    model: str = None,
    source_lang: str = 'hi',
    target_lang: str = 'mwr',
    texts: List[str] = None
) -> Dict:
    """
    Test a single provider.

    Args:
        provider_name: Provider name
        model: Model name (uses default if None)
        source_lang: Source language code
        target_lang: Target language code
        texts: List of texts to translate

    Returns:
        Test results dictionary
    """
    if texts is None:
        texts = TEST_SENTENCES

    print(f"\n{'='*70}")
    print(f"Testing: {provider_name.upper()}")
    print(f"{'='*70}")

    # Check if provider is available
    if not ProviderRegistry.check_provider_available(provider_name):
        env_var = ProviderRegistry.API_KEY_ENV_VARS.get(provider_name)
        print(f"❌ SKIPPED: {env_var} not set in environment")
        return {
            'provider': provider_name,
            'status': 'skipped',
            'reason': f'{env_var} not set'
        }

    # Get model info
    if model is None:
        model = ProviderRegistry.get_default_model(provider_name)

    print(f"Provider: {provider_name}")
    print(f"Model: {model}")
    print(f"Language pair: {source_lang} → {target_lang}")

    # Create translator
    try:
        translator = create_translator(
            provider=provider_name,
            model=model,
            source_lang=source_lang,
            target_lang=target_lang
        )
        translator.initialize()
        print(f"✓ Translator initialized")

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return {
            'provider': provider_name,
            'model': model,
            'status': 'failed',
            'error': str(e)
        }

    # Translate
    results = {
        'provider': provider_name,
        'model': model,
        'status': 'success',
        'translations': [],
        'total_cost': 0.0,
        'successful': 0,
        'failed': 0
    }

    for i, text in enumerate(texts):
        print(f"\n[{i+1}/{len(texts)}] Source: {text}")

        # Create translation request
        request = TranslationRequest(
            units=[TranslationUnit(text=text, index=0)],
            src_lang=source_lang,
            tgt_lang=target_lang,
            parameters={'temperature': 0.3, 'max_tokens': 2048}
        )

        try:
            response = translator.translate_sync(request)

            if response.results[0].status == TranslationStatus.COMPLETED:
                translation = response.results[0].translation
                cost = response.results[0].cost
                print(f"       Target: {translation}")
                print(f"       Cost: ₹{cost:.6f} INR (${cost/85:.6f} USD)")

                results['translations'].append({
                    'source': text,
                    'translation': translation,
                    'cost': cost
                })
                results['successful'] += 1
                results['total_cost'] += cost
            else:
                error = response.results[0].error
                print(f"       ❌ Failed: {error}")
                results['failed'] += 1

        except Exception as e:
            print(f"       ❌ Error: {e}")
            results['failed'] += 1

    # Print summary
    print(f"\n{'-'*70}")
    print(f"Summary for {provider_name} ({model}):")
    print(f"  Successful: {results['successful']}/{len(texts)}")
    print(f"  Failed: {results['failed']}/{len(texts)}")
    print(f"  Total cost: ₹{results['total_cost']:.6f} INR (${results['total_cost']/85:.6f} USD)")
    if results['successful'] > 0:
        avg_cost = results['total_cost'] / results['successful']
        print(f"  Avg cost per sentence: ₹{avg_cost:.6f} (${avg_cost/85:.6f} USD)")
    print(f"{'-'*70}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Test translation with all available providers",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--provider',
        choices=['all'] + ProviderRegistry.list_providers(),
        default='all',
        help='Provider to test (default: all available)'
    )
    parser.add_argument(
        '--model',
        help='Model to use (uses provider default if not specified)'
    )
    parser.add_argument(
        '--source-lang',
        default='hi',
        help='Source language code (default: hi)'
    )
    parser.add_argument(
        '--target-lang',
        default='mwr',
        help='Target language code (default: mwr)'
    )
    parser.add_argument(
        '--text',
        help='Single text to translate (overrides default test sentences)'
    )
    parser.add_argument(
        '--list-available',
        action='store_true',
        help='List available providers and exit'
    )

    args = parser.parse_args()

    # List available providers
    if args.list_available:
        print("Available providers (with API keys configured):")
        for provider in ProviderRegistry.list_available_providers():
            default_model = ProviderRegistry.get_default_model(provider)
            print(f"  ✓ {provider:12s} (default: {default_model})")

        print("\nProviders requiring setup:")
        for provider in ProviderRegistry.list_providers():
            if not ProviderRegistry.check_provider_available(provider):
                env_var = ProviderRegistry.API_KEY_ENV_VARS.get(provider)
                print(f"  ✗ {provider:12s} (set {env_var})")
        return

    # Get test texts
    texts = [args.text] if args.text else TEST_SENTENCES

    # Determine which providers to test
    if args.provider == 'all':
        providers = ProviderRegistry.list_available_providers()
        if not providers:
            print("❌ No providers available. Please set API keys in .env file.")
            print("\nSee API_KEYS_SETUP.md for instructions.")
            return
        print(f"Testing {len(providers)} available provider(s): {', '.join(providers)}")
    else:
        providers = [args.provider]

    # Test each provider
    all_results = []
    for provider in providers:
        result = test_provider(
            provider_name=provider,
            model=args.model,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            texts=texts
        )
        all_results.append(result)

    # Print overall summary
    print(f"\n\n{'='*70}")
    print("OVERALL SUMMARY")
    print(f"{'='*70}")

    successful_providers = [r for r in all_results if r['status'] == 'success']
    failed_providers = [r for r in all_results if r['status'] == 'failed']
    skipped_providers = [r for r in all_results if r['status'] == 'skipped']

    print(f"\nProviders tested: {len(all_results)}")
    print(f"  ✓ Successful: {len(successful_providers)}")
    print(f"  ❌ Failed: {len(failed_providers)}")
    print(f"  ⊘ Skipped: {len(skipped_providers)}")

    if successful_providers:
        print("\n--- Cost Comparison ---")
        # Sort by total cost
        successful_providers.sort(key=lambda x: x.get('total_cost', float('inf')))

        for result in successful_providers:
            provider = result['provider']
            model = result['model']
            total_cost = result['total_cost']
            success_rate = result['successful'] / (result['successful'] + result['failed']) * 100

            print(f"\n{provider} ({model}):")
            print(f"  Total cost: ₹{total_cost:.6f} INR (${total_cost/85:.6f} USD)")
            print(f"  Success rate: {success_rate:.1f}%")

            if result['successful'] > 0:
                avg_cost = total_cost / result['successful']
                print(f"  Avg per sentence: ₹{avg_cost:.6f} (${avg_cost/85:.6f} USD)")

    if failed_providers:
        print("\n--- Failed Providers ---")
        for result in failed_providers:
            print(f"  ❌ {result['provider']} ({result['model']}): {result.get('error', 'Unknown error')}")

    if skipped_providers:
        print("\n--- Skipped Providers ---")
        for result in skipped_providers:
            print(f"  ⊘ {result['provider']}: {result.get('reason', 'Unknown reason')}")

    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    main()
