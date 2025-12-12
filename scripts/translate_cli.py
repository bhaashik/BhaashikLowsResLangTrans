#!/usr/bin/env python3
"""
Unified CLI tool for translation across multiple LLM providers.

Usage:
    python scripts/translate_cli.py \\
        --provider openai \\
        --model gpt-4o-mini \\
        --source-lang hi \\
        --target-lang mwr \\
        --input input.txt \\
        --output output.txt

    # Or with prompt config
    python scripts/translate_cli.py \\
        --provider anthropic \\
        --source-lang hi \\
        --target-lang bra \\
        --prompt-config universal_translate/config/prompts/hi_to_bra_claude.yaml \\
        --input texts.txt
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from universal_translate.providers.registry import ProviderRegistry, create_translator
from universal_translate.core import (
    TranslationUnit,
    TranslationRequest,
    TranslationStatus
)


def read_input_file(file_path: str) -> List[str]:
    """Read input texts from file (one per line)."""
    path = Path(file_path)
    if not path.exists():
        print(f"Error: Input file not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    with open(path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    return lines


def write_output_file(file_path: str, translations: List[str]):
    """Write translations to file (one per line)."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        for translation in translations:
            f.write(translation + '\n')


def print_cost_summary(response):
    """Print cost summary from translation response."""
    print("\n" + "=" * 60)
    print("COST SUMMARY")
    print("=" * 60)
    print(f"Provider: {response.provider}")
    print(f"Model: {response.model}")
    print(f"Total cost: ₹{response.total_cost:.4f} INR (${response.total_cost/85:.4f} USD)")

    # Calculate success rate
    successful = sum(1 for r in response.results if r.status == TranslationStatus.COMPLETED)
    total = len(response.results)
    print(f"Success rate: {successful}/{total} ({successful/total*100:.1f}%)")

    # Show per-sentence cost
    if successful > 0:
        avg_cost = response.total_cost / successful
        print(f"Average cost per sentence: ₹{avg_cost:.6f} ({avg_cost/85:.6f} USD)")

    # Show caching info if available
    if response.metadata.get('use_caching'):
        cached_count = sum(1 for r in response.results
                          if r.metadata and r.metadata.get('cached_tokens', 0) > 0)
        if cached_count > 0:
            print(f"Cache hits: {cached_count}/{total}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Unified translation CLI supporting multiple LLM providers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Translate with OpenAI GPT-4o-mini
  python scripts/translate_cli.py \\
      --provider openai \\
      --source-lang hi --target-lang mwr \\
      --input input.txt --output output.txt

  # Translate with Anthropic Claude
  python scripts/translate_cli.py \\
      --provider anthropic --model claude-haiku-4.5 \\
      --source-lang hi --target-lang bra \\
      --input input.txt

  # Translate with Gemini (free tier)
  python scripts/translate_cli.py \\
      --provider gemini --model gemini-2.0-flash-exp \\
      --source-lang hi --target-lang awa \\
      --text "मैं घर जा रहा हूं।"

  # Use custom prompt configuration
  python scripts/translate_cli.py \\
      --provider openai \\
      --prompt-config universal_translate/config/prompts/hi_to_mwr_openai.yaml \\
      --input input.txt

  # List available providers and models
  python scripts/translate_cli.py --list-providers
  python scripts/translate_cli.py --list-models openai
        """
    )

    # Provider and model selection
    parser.add_argument(
        '--provider',
        choices=ProviderRegistry.list_providers(),
        help='LLM provider to use'
    )
    parser.add_argument(
        '--model',
        help='Model name (uses provider default if not specified)'
    )

    # Language pair
    parser.add_argument(
        '--source-lang',
        default='hi',
        help='Source language code (default: hi)'
    )
    parser.add_argument(
        '--target-lang',
        help='Target language code (e.g., mwr, awa, bra, mai, nep)'
    )

    # Input/output
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        '--input',
        help='Input file (one sentence per line)'
    )
    input_group.add_argument(
        '--text',
        help='Single text to translate'
    )

    parser.add_argument(
        '--output',
        help='Output file (default: stdout)'
    )

    # Configuration
    parser.add_argument(
        '--prompt-config',
        help='Path to prompt configuration YAML file'
    )
    parser.add_argument(
        '--api-key',
        help='API key (uses environment variable if not specified)'
    )

    # Translation parameters
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.3,
        help='Temperature for generation (default: 0.3)'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=2048,
        help='Maximum tokens for output (default: 2048)'
    )

    # Batch processing
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10,
        help='Number of sentences to process at once (default: 10)'
    )

    # Display options
    parser.add_argument(
        '--show-cost',
        action='store_true',
        help='Show cost summary'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress messages'
    )

    # Information commands
    parser.add_argument(
        '--list-providers',
        action='store_true',
        help='List available providers'
    )
    parser.add_argument(
        '--list-models',
        metavar='PROVIDER',
        help='List available models for a provider'
    )

    args = parser.parse_args()

    # Handle information commands
    if args.list_providers:
        print("Available providers:")
        for provider in ProviderRegistry.list_providers():
            available = "✓" if ProviderRegistry.check_provider_available(provider) else "✗"
            default_model = ProviderRegistry.get_default_model(provider)
            env_var = ProviderRegistry.API_KEY_ENV_VARS.get(provider, 'N/A')
            print(f"  [{available}] {provider:12s} (default: {default_model}, env: {env_var})")
        return

    if args.list_models:
        models = ProviderRegistry.list_models(args.list_models)
        print(f"Available models for {args.list_models}:")
        for model in models:
            print(f"  - {model}")
        return

    # Validate required arguments
    if not args.provider:
        # Try to auto-detect
        args.provider = ProviderRegistry.auto_detect_provider()
        if not args.provider:
            parser.error("--provider required (or set API key environment variable)")

    if not args.target_lang:
        parser.error("--target-lang required")

    if not args.input and not args.text:
        parser.error("--input or --text required")

    # Create translator
    if not args.quiet:
        print(f"Creating {args.provider} translator...")
        if args.model:
            print(f"Model: {args.model}")
        else:
            print(f"Model: {ProviderRegistry.get_default_model(args.provider)} (default)")

    try:
        translator = create_translator(
            provider=args.provider,
            model=args.model,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            prompt_config=args.prompt_config,
            api_key=args.api_key
        )

        # Initialize translator
        translator.initialize()

        if not args.quiet:
            print(f"Translator initialized successfully.")

    except Exception as e:
        print(f"Error creating translator: {e}", file=sys.stderr)
        sys.exit(1)

    # Get input texts
    if args.text:
        texts = [args.text]
    else:
        texts = read_input_file(args.input)
        if not args.quiet:
            print(f"Loaded {len(texts)} sentences from {args.input}")

    # Create translation request
    units = [
        TranslationUnit(text=text, index=i)
        for i, text in enumerate(texts)
    ]

    request = TranslationRequest(
        units=units,
        src_lang=args.source_lang,
        tgt_lang=args.target_lang,
        parameters={
            'temperature': args.temperature,
            'max_tokens': args.max_tokens
        }
    )

    # Translate
    if not args.quiet:
        print(f"Translating {len(texts)} sentence(s)...")

    try:
        response = translator.translate_sync(request)
    except Exception as e:
        print(f"Error during translation: {e}", file=sys.stderr)
        sys.exit(1)

    # Extract translations
    translations = []
    for result in response.results:
        if result.status == TranslationStatus.COMPLETED:
            translations.append(result.translation)
        else:
            print(f"Warning: Translation failed for: {result.source[:50]}...", file=sys.stderr)
            print(f"  Error: {result.error}", file=sys.stderr)
            translations.append("")  # Empty translation for failed items

    # Output translations
    if args.output:
        write_output_file(args.output, translations)
        if not args.quiet:
            print(f"Translations written to {args.output}")
    else:
        # Print to stdout
        for i, translation in enumerate(translations):
            if not args.quiet and len(texts) > 1:
                print(f"\n[{i+1}] {texts[i]}")
                print(f"→   {translation}")
            else:
                print(translation)

    # Show cost summary if requested
    if args.show_cost or not args.quiet:
        print_cost_summary(response)


if __name__ == '__main__':
    main()
