#!/usr/bin/env python3
"""
Example 4: Quality Assessment and Comparison

This example demonstrates how to assess and compare translation quality
using various metrics.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.translators import IndicTrans2Translator, NLLBTranslator
from src.quality import QualityMetrics
from src.utils.logger import setup_logger


def main():
    # Setup logging
    setup_logger(level='INFO', console=True)

    print("="*70)
    print("EXAMPLE 4: QUALITY ASSESSMENT")
    print("="*70)

    # Sample data (English-Hindi)
    sources = [
        "Hello, how are you?",
        "What is your name?",
        "I am learning Hindi.",
        "Thank you very much!",
        "Have a great day!",
        "Where is the nearest hospital?",
        "Can you help me?",
        "I would like some water.",
        "How much does this cost?",
        "I am very happy to meet you."
    ]

    # Reference translations (high-quality Hindi)
    references = [
        "नमस्ते, आप कैसे हैं?",
        "आपका नाम क्या है?",
        "मैं हिंदी सीख रहा हूं।",
        "बहुत बहुत धन्यवाद!",
        "आपका दिन शुभ हो!",
        "निकटतम अस्पताल कहाँ है?",
        "क्या आप मेरी मदद कर सकते हैं?",
        "मुझे थोड़ा पानी चाहिए।",
        "इसकी कीमत कितनी है?",
        "मैं आपसे मिलकर बहुत खुश हूं।"
    ]

    # Example 1: Compare Translation Systems
    print("\n" + "-"*70)
    print("Example 1: Compare IndicTrans2 vs NLLB")
    print("-"*70)

    print("\nTranslating with IndicTrans2...")
    it_translator = IndicTrans2Translator()
    translations_it = it_translator.translate(
        sources,
        src_lang='en',
        tgt_lang='hi',
        show_progress=False
    )

    print("Translating with NLLB...")
    nllb_translator = NLLBTranslator()
    translations_nllb = nllb_translator.translate(
        sources,
        src_lang='en',
        tgt_lang='hi',
        show_progress=False
    )

    # Calculate metrics
    print("\nCalculating quality metrics...")
    metrics = QualityMetrics()

    comparison = metrics.compare_translations(
        {
            'IndicTrans2': translations_it,
            'NLLB': translations_nllb
        },
        references
    )

    # Print results
    print("\n" + "="*70)
    print("QUALITY COMPARISON")
    print("="*70)

    for system, scores in comparison.items():
        metrics.print_metrics(scores, f"{system} Results")

    # Example 2: Detailed Analysis
    print("\n" + "-"*70)
    print("Example 2: Detailed Translation Analysis")
    print("-"*70)

    print("\nSample translations with scores:")
    for i in range(min(3, len(sources))):
        print(f"\n[{i+1}]")
        print(f"  Source:     {sources[i]}")
        print(f"  Reference:  {references[i]}")
        print(f"  IndicTrans2: {translations_it[i]}")
        print(f"  NLLB:       {translations_nllb[i]}")

    # Example 3: Quality-based Tier Assignment
    print("\n" + "-"*70)
    print("Example 3: Quality Assessment for Tier Assignment")
    print("-"*70)

    # Assess IndicTrans2 quality
    quality_assessment = metrics.assess_quality_tier(
        translations_it,
        references,
        threshold_high=40.0,
        threshold_medium=25.0
    )

    print(f"\nQuality Tier: {quality_assessment['quality_tier'].upper()}")
    print(f"Description: {quality_assessment['description']}")
    print(f"BLEU Score: {quality_assessment['bleu_score']:.2f}")

    print("\nRecommendation based on quality:")
    if quality_assessment['quality_tier'] == 'high':
        print("  → IndicTrans2 quality is HIGH")
        print("  → Use 90% free, 10% API enhancement")
        print("  → Expected cost: ~₹5,000 per 1M pairs")
    elif quality_assessment['quality_tier'] == 'medium':
        print("  → IndicTrans2 quality is MEDIUM")
        print("  → Use 70% free, 30% API enhancement")
        print("  → Expected cost: ~₹15,000 per 1M pairs")
    else:
        print("  → IndicTrans2 quality is LOW")
        print("  → Use 50% free, 50% API enhancement")
        print("  → Expected cost: ~₹25,000 per 1M pairs")

    # Example 4: Sampling for Manual Review
    print("\n" + "-"*70)
    print("Example 4: Sample for Manual Review")
    print("-"*70)

    samples = metrics.sample_for_review(
        sources=sources,
        translations=translations_it,
        references=references,
        sample_rate=0.5,  # 50% for demo
        min_samples=3,
        max_samples=5
    )

    print(f"\nSampled {len(samples)} translations for review:")
    for idx, src, trans, ref in samples:
        print(f"\n[Index: {idx}]")
        print(f"  Source:      {src}")
        print(f"  Translation: {trans}")
        print(f"  Reference:   {ref}")
        print(f"  Match: {'✓' if trans == ref else '✗'}")

    # Best Practices
    print("\n" + "-"*70)
    print("BEST PRACTICES FOR QUALITY ASSESSMENT")
    print("-"*70)
    print("""
    1. Always compare multiple systems on YOUR data:
       - Different models perform differently on different domains
       - Your data characteristics matter (length, complexity, domain)

    2. Use multiple metrics:
       - BLEU: Good for overall quality
       - chrF: Better for morphologically rich languages
       - TER: Measures edit distance
       - Human evaluation: Always best for final decision

    3. Sample-based evaluation:
       - Evaluate on 1000-5000 samples minimum
       - Stratify samples (different lengths, complexities)
       - Include domain-specific content

    4. Quality-driven tier assignment:
       - High quality (BLEU > 40): Use 90% free
       - Medium quality (BLEU 25-40): Use 70% free
       - Low quality (BLEU < 25): Use 50% free or consider alternatives

    5. Continuous monitoring:
       - Track quality over time
       - Re-evaluate as models improve
       - Adjust strategies based on feedback
    """)

    print("\n" + "="*70)
    print("EXAMPLE COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
