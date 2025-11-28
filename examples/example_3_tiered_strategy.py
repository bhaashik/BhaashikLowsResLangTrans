#!/usr/bin/env python3
"""
Example 3: Tiered Translation Strategy (Cost-Optimized)

This example demonstrates the tiered approach that optimally combines
free and paid translation methods for best quality at minimal cost.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tiered_orchestrator import TieredOrchestrator
from src.utils.logger import setup_logger


def main():
    # Setup logging
    setup_logger(level='INFO', console=True)

    print("="*70)
    print("EXAMPLE 3: TIERED TRANSLATION STRATEGY")
    print("="*70)

    # Example texts
    texts = [
        "Hello, how are you?",
        "What is your name?",
        "I am learning a new language.",
        "This is a beautiful day.",
        "Thank you very much for your help!",
        "Can you please help me?",
        "I would like to order some food.",
        "Where is the nearest hospital?",
        "How much does this cost?",
        "I am very happy to meet you."
    ]

    # Example 1: Cost Estimation
    print("\n" + "-"*70)
    print("Example 1: Cost Estimation")
    print("-"*70)

    print("\nInitializing orchestrator...")
    orchestrator = TieredOrchestrator()

    print("\nEstimating cost for 10,000 samples...")
    orchestrator.print_cost_estimate(
        num_samples=10000,
        avg_tokens_per_sample=100
    )

    # Example 2: Default Tiered Strategy (70/20/10)
    print("\n" + "-"*70)
    print("Example 2: Default Strategy (70/20/10)")
    print("-"*70)
    print("""
    This strategy uses:
    - 70% Free Hindi pivot (IndicTrans2)
    - 20% Claude Haiku 3.5 (quality enhancement)
    - 10% Claude Haiku 4.5 (premium quality)
    """)

    print("\n⚠️  WARNING: This will use API credits!")
    print("To run this example, you need to:")
    print("1. Set ANTHROPIC_API_KEY in .env file")
    print("2. Uncomment the translation code below")
    print("3. Be aware of costs (~₹0.05 per sample for this mix)\n")

    # Uncomment to actually run (costs money!)
    """
    results = orchestrator.translate(
        texts,
        src_lang='en',
        tgt_lang='bho',
        seed=42,
        save_results=True
    )

    print("\nResults:")
    for i, (src, trans, tier) in enumerate(zip(
        results['sources'][:5],
        results['translations'][:5],
        results['tier_assignments'][:5]
    ), 1):
        print(f"\n[{i}] Tier: {tier}")
        print(f"  Source: {src}")
        print(f"  Translation: {trans}")

    print(f"\nCost Summary:")
    print(f"  Total Cost: {results['cost_summary']['currency']} "
          f"{results['cost_summary']['total_cost']:.2f}")
    """

    # Example 3: Custom Tier Distribution
    print("\n" + "-"*70)
    print("Example 3: Custom Distribution (90/8/2)")
    print("-"*70)

    print("\nCustom strategy: 90% free, 8% enhancement, 2% premium")

    custom_orchestrator = TieredOrchestrator(
        tier_percentages={
            'free': 90,
            'enhancement': 8,
            'premium': 2
        }
    )

    print("\nCost estimate for 10,000 samples:")
    custom_orchestrator.print_cost_estimate(
        num_samples=10000,
        avg_tokens_per_sample=100
    )

    # Example 4: Pure Free Strategy
    print("\n" + "-"*70)
    print("Example 4: Pure Free Strategy (100/0/0)")
    print("-"*70)

    print("\nPure free strategy: 100% Hindi pivot, 0% API")

    free_orchestrator = TieredOrchestrator(
        tier_percentages={
            'free': 100,
            'enhancement': 0,
            'premium': 0
        }
    )

    print("\nThis strategy is completely FREE! ✓")
    print("Cost: ₹0 (compute only)")

    # You can safely run this one (no API costs)
    print("\nRunning free translation on sample...")
    results_free = free_orchestrator.translate(
        texts[:3],  # Just 3 samples
        src_lang='en',
        tgt_lang='bho',
        show_progress=False,
        save_results=False
    )

    print("\nResults:")
    for i, (src, trans) in enumerate(zip(
        results_free['sources'],
        results_free['translations']
    ), 1):
        print(f"\n[{i}]")
        print(f"  English: {src}")
        print(f"  Bhojpuri: {trans}")

    # Recommendations
    print("\n" + "-"*70)
    print("RECOMMENDATIONS")
    print("-"*70)
    print("""
    1. Start with 100% free strategy:
       - Test on sample data
       - Assess quality manually
       - Identify problematic cases

    2. Gradually introduce API enhancement:
       - Start with 90/10/0 (90% free, 10% enhancement)
       - Monitor quality improvement
       - Adjust percentages based on results

    3. Optimal production setup:
       - 70/20/10 for balanced quality/cost
       - 85/10/5 for cost-conscious
       - 60/25/15 for quality-focused

    4. Budget planning:
       - Per language (2M pairs): ₹10,485 (~$123) with 70/20/10
       - All 6 languages: ₹62,912 (~$740) with optimizations
       - Use batch API + prompt caching for 30-40% savings

    5. Quality validation:
       - Always validate on sample (~1000 pairs)
       - Use BLEU scores to compare strategies
       - Consider human evaluation for critical content
    """)

    print("\n" + "="*70)
    print("EXAMPLE COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
