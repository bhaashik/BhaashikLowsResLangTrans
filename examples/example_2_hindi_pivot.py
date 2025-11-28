#!/usr/bin/env python3
"""
Example 2: Hindi Pivot Translation for Unsupported Languages

This example demonstrates translation for languages not directly supported
by IndicTrans2 using Hindi as a pivot language.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.translators import HindiPivotTranslator
from src.utils.logger import setup_logger


def main():
    # Setup logging
    setup_logger(level='INFO', console=True)

    print("="*70)
    print("EXAMPLE 2: HINDI PIVOT TRANSLATION")
    print("="*70)

    # Initialize translator
    print("\nInitializing Hindi pivot translator...")
    translator = HindiPivotTranslator()

    # Show supported pivot languages
    print("\nSupported Hindi pivot languages:")
    pivot_langs = translator.get_pivot_languages()
    for code, name in pivot_langs.items():
        info = translator.get_language_info(code)
        speakers = info.get('speakers', 'Unknown')
        print(f"  {code}: {name} (~{speakers:,} speakers)")

    # Example 1: English to Bhojpuri (via Hindi)
    print("\n" + "-"*70)
    print("Example 1: English to Bhojpuri (via Hindi pivot)")
    print("-"*70)

    texts_en = [
        "Hello, how are you?",
        "What is your name?",
        "I am learning Bhojpuri.",
        "Thank you very much!",
        "Have a great day!"
    ]

    print("\nTranslating English to Bhojpuri...")
    print("(Note: Using Hindi as approximation for Bhojpuri)")

    result = translator.translate_via_hindi(
        texts_en,
        src_lang='en',
        tgt_lang='bho',
        return_intermediate=True,
        show_progress=False
    )

    print("\nResults:")
    for i, (src, hindi, final) in enumerate(zip(texts_en,
                                                  result['intermediate_hindi'],
                                                  result['translations']), 1):
        print(f"\n[{i}]")
        print(f"  English: {src}")
        print(f"  Hindi (intermediate): {hindi}")
        print(f"  Bhojpuri (approximation): {final}")

    # Example 2: Batch processing for multiple pivot languages
    print("\n" + "-"*70)
    print("Example 2: English to Multiple Pivot Languages")
    print("-"*70)

    text = "Hello, how are you?"
    print(f"\nSource: {text}")

    # Translate to different pivot languages
    for lang_code, lang_name in list(pivot_langs.items())[:3]:  # First 3 languages
        print(f"\nTranslating to {lang_name} ({lang_code})...")
        translation = translator.translate(
            text,
            src_lang='en',
            tgt_lang=lang_code,
            show_progress=False
        )
        print(f"  Result: {translation}")

    # Note about quality
    print("\n" + "-"*70)
    print("IMPORTANT NOTES")
    print("-"*70)
    print("""
    1. Hindi pivot provides FREE translation but with limitations:
       - Uses Hindi as approximation for linguistically close languages
       - Quality depends on linguistic similarity to Hindi
       - Best for Indo-Aryan languages (Bhojpuri, Magahi, Awadhi, etc.)

    2. For better quality:
       - Use the tiered strategy (combines free + API)
       - Consider post-editing by native speakers
       - Use Claude API for final polishing

    3. Recommended workflow:
       - Start with Hindi pivot (free)
       - Assess quality on sample
       - Enhance with API if needed (10-30% of data)
    """)

    print("\n" + "="*70)
    print("EXAMPLE COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
