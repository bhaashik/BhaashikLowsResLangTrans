#!/usr/bin/env python3
"""
Example 1: Basic Translation using IndicTrans2

This example demonstrates simple translation between supported languages.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.translators import IndicTrans2Translator
from src.utils.logger import setup_logger


def main():
    # Setup logging
    setup_logger(level='INFO', console=True)

    print("="*70)
    print("EXAMPLE 1: BASIC TRANSLATION WITH INDICTRANS2")
    print("="*70)

    # Initialize translator
    print("\nInitializing translator...")
    translator = IndicTrans2Translator()

    # Example 1: Single text translation
    print("\n" + "-"*70)
    print("Example 1: English to Hindi")
    print("-"*70)

    text_en = "Hello, how are you? I hope you are doing well."
    print(f"Source (English): {text_en}")

    translation_hi = translator.translate(
        text_en,
        src_lang='en',
        tgt_lang='hi'
    )
    print(f"Translation (Hindi): {translation_hi}")

    # Example 2: Batch translation
    print("\n" + "-"*70)
    print("Example 2: Batch Translation - English to Multiple Languages")
    print("-"*70)

    texts = [
        "Hello, how are you?",
        "What is your name?",
        "I am learning a new language.",
        "Thank you very much!",
        "Have a great day!"
    ]

    target_languages = ['hi', 'bn', 'ta', 'te']

    for tgt_lang in target_languages:
        print(f"\nTranslating to {tgt_lang.upper()}...")
        translations = translator.translate_batch(
            texts,
            src_lang='en',
            tgt_lang=tgt_lang,
            show_progress=False
        )

        print(f"Sample: {texts[0]} → {translations[0]}")

    # Example 3: Indic to Indic translation
    print("\n" + "-"*70)
    print("Example 3: Indic to Indic - Hindi to Bengali")
    print("-"*70)

    text_hi = "आप कैसे हैं? मुझे आशा है कि आप अच्छे हैं।"
    print(f"Source (Hindi): {text_hi}")

    translation_bn = translator.translate(
        text_hi,
        src_lang='hi',
        tgt_lang='bn'
    )
    print(f"Translation (Bengali): {translation_bn}")

    print("\n" + "="*70)
    print("EXAMPLE COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
