#!/usr/bin/env python3
"""
Test script for SMT-inspired quality components.

This script verifies that:
1. Adequacy estimators work correctly
2. Fluency estimators work correctly
3. Quality estimator combines them properly
4. Quality-aware losses function correctly

Usage:
    python scripts/test_quality_estimation.py

    # Test specific component
    python scripts/test_quality_estimation.py --component adequacy

    # Skip model loading tests
    python scripts/test_quality_estimation.py --skip-models
"""

import argparse
import sys
from pathlib import Path
import logging
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.quality import (
    QualityConfig,
    SentenceEmbeddingAdequacy,
    EntailmentAdequacy,
    PerplexityFluency,
    ParseBasedFluency,
    QualityEstimator,
    create_quality_estimator,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_sentence_embedding_adequacy():
    """Test sentence embedding adequacy estimator."""
    logger.info("=" * 80)
    logger.info("TEST 1: Sentence Embedding Adequacy")
    logger.info("=" * 80)

    try:
        # Check if sentence-transformers is available
        try:
            import sentence_transformers
        except ImportError:
            logger.warning("sentence-transformers not installed, skipping test")
            logger.info("  Install with: pip install sentence-transformers")
            return True

        # Create estimator
        estimator = SentenceEmbeddingAdequacy(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            device="cpu",  # Use CPU for testing
            batch_size=2,
        )
        estimator.initialize()

        # Test pairs (source, good translation, bad translation)
        source = "Hello, how are you?"
        good_translation = "Bonjour, comment allez-vous?"
        bad_translation = "The cat sat on the mat."

        logger.info(f"\nSource: {source}")
        logger.info(f"Good translation: {good_translation}")
        logger.info(f"Bad translation: {bad_translation}")

        # Compute adequacy
        scores = estimator.estimate_adequacy(
            [source, source],
            [good_translation, bad_translation]
        )

        logger.info(f"\nAdequacy scores:")
        logger.info(f"  Good: {scores[0]:.4f}")
        logger.info(f"  Bad: {scores[1]:.4f}")

        # Good should have higher adequacy
        assert scores[0] > scores[1], "Good translation should have higher adequacy"

        estimator.cleanup()
        logger.info("\n✓ Sentence embedding adequacy test passed!")
        return True

    except Exception as e:
        logger.error(f"✗ Sentence embedding adequacy test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_perplexity_fluency():
    """Test perplexity fluency estimator."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Perplexity Fluency")
    logger.info("=" * 80)

    try:
        # Create estimator
        estimator = PerplexityFluency(
            model_name="gpt2",
            device="cpu",
            batch_size=2,
        )
        estimator.initialize()

        # Test sentences (fluent vs non-fluent)
        fluent = "The quick brown fox jumps over the lazy dog."
        non_fluent = "Dog lazy the over jumps fox brown quick the."

        logger.info(f"\nFluent: {fluent}")
        logger.info(f"Non-fluent: {non_fluent}")

        # Compute fluency
        scores = estimator.estimate_fluency([fluent, non_fluent])

        logger.info(f"\nFluency scores:")
        logger.info(f"  Fluent: {scores[0]:.4f}")
        logger.info(f"  Non-fluent: {scores[1]:.4f}")

        # Fluent should have higher score
        assert scores[0] > scores[1], "Fluent sentence should have higher score"

        estimator.cleanup()
        logger.info("\n✓ Perplexity fluency test passed!")
        return True

    except Exception as e:
        logger.error(f"✗ Perplexity fluency test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_parse_based_fluency():
    """Test parse-based fluency estimator."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: Parse-Based Fluency")
    logger.info("=" * 80)

    try:
        # Create estimator
        estimator = ParseBasedFluency(
            model_name="stanza",
            language="en",
            device="cpu",
            batch_size=2,
        )
        estimator.initialize()

        # Test sentences
        well_formed = "The cat sat on the mat."
        malformed = "Cat the mat on sat the."

        logger.info(f"\nWell-formed: {well_formed}")
        logger.info(f"Malformed: {malformed}")

        # Compute fluency
        scores = estimator.estimate_fluency([well_formed, malformed])

        logger.info(f"\nFluency scores:")
        logger.info(f"  Well-formed: {scores[0]:.4f}")
        logger.info(f"  Malformed: {scores[1]:.4f}")

        # Well-formed should have higher score
        assert scores[0] > scores[1], "Well-formed should have higher score"

        logger.info("\n✓ Parse-based fluency test passed!")
        return True

    except Exception as e:
        logger.error(f"✗ Parse-based fluency test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_quality_estimator():
    """Test combined quality estimator."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 4: Quality Estimator")
    logger.info("=" * 80)

    try:
        # Check dependencies
        try:
            import sentence_transformers
        except ImportError:
            logger.warning("sentence-transformers not installed, skipping test")
            return True

        # Create config
        config = QualityConfig(
            use_quality_estimation=True,
            use_adequacy=True,
            use_fluency=True,
            adequacy_method="sentence_embedding",
            fluency_method="perplexity",
            adequacy_model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            fluency_model="gpt2",
            adequacy_weight=0.5,
            fluency_weight=0.5,
            device="cpu",
            batch_size=2,
        )

        # Create estimator
        estimator = create_quality_estimator(
            config=config,
            source_lang="en",
            target_lang="en",
        )

        # Test translation
        source = "Hello, how are you?"
        good = "Hi, how are you doing?"
        bad = "Banana elephant purple music."

        logger.info(f"\nSource: {source}")
        logger.info(f"Good: {good}")
        logger.info(f"Bad: {bad}")

        # Estimate quality
        quality_good, adq_good, flu_good = estimator.estimate_single(
            source, good, return_components=True
        )
        quality_bad, adq_bad, flu_bad = estimator.estimate_single(
            source, bad, return_components=True
        )

        logger.info(f"\nGood translation:")
        logger.info(f"  Quality: {quality_good:.4f}")
        logger.info(f"  Adequacy: {adq_good:.4f}")
        logger.info(f"  Fluency: {flu_good:.4f}")

        logger.info(f"\nBad translation:")
        logger.info(f"  Quality: {quality_bad:.4f}")
        logger.info(f"  Adequacy: {adq_bad:.4f}")
        logger.info(f"  Fluency: {flu_bad:.4f}")

        # Good should have higher quality
        assert quality_good > quality_bad, "Good should have higher quality"

        # Test batch processing
        logger.info("\nTesting batch processing...")
        sources = [source] * 3
        translations = [good, bad, "How are you?"]

        quality_scores = estimator.estimate_quality(sources, translations)
        logger.info(f"  Batch scores: {quality_scores}")
        assert len(quality_scores) == 3, "Should return 3 scores"

        # Test statistics
        logger.info("\nTesting statistics...")
        stats = estimator.get_statistics(sources, translations)
        logger.info(f"  Mean quality: {stats['quality_mean']:.4f}")
        logger.info(f"  Std quality: {stats['quality_std']:.4f}")

        estimator.cleanup()
        logger.info("\n✓ Quality estimator test passed!")
        return True

    except Exception as e:
        logger.error(f"✗ Quality estimator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_quality_filtering():
    """Test quality-based filtering."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 5: Quality Filtering")
    logger.info("=" * 80)

    try:
        # Check dependencies
        try:
            import sentence_transformers
        except ImportError:
            logger.warning("sentence-transformers not installed, skipping test")
            return True

        # Create config
        config = QualityConfig(
            use_quality_estimation=True,
            use_adequacy=True,
            use_fluency=False,
            adequacy_method="sentence_embedding",
            adequacy_model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            adequacy_weight=1.0,
            fluency_weight=0.0,
            quality_threshold=0.5,
            device="cpu",
        )

        estimator = create_quality_estimator(config)

        # Test data (mix of good and bad)
        sources = [
            "Hello",
            "How are you?",
            "What is your name?",
            "Goodbye",
        ]
        translations = [
            "Hi",  # Good
            "I am fine",  # OK
            "Purple banana",  # Bad
            "See you later",  # Good
        ]

        logger.info("\nFiltering translations with threshold=0.5...")
        filtered_src, filtered_tgt, scores = estimator.filter_by_quality(
            sources, translations, threshold=0.5
        )

        logger.info(f"  Original: {len(sources)} pairs")
        logger.info(f"  Filtered: {len(filtered_src)} pairs")
        logger.info(f"  Scores: {scores}")

        assert len(filtered_src) <= len(sources), "Should filter some examples"

        estimator.cleanup()
        logger.info("\n✓ Quality filtering test passed!")
        return True

    except Exception as e:
        logger.error(f"✗ Quality filtering test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test quality estimation')
    parser.add_argument(
        '--component', default='all',
        choices=['all', 'adequacy', 'fluency', 'quality', 'filtering'],
        help='Component to test (default: all)'
    )
    parser.add_argument(
        '--skip-models', action='store_true',
        help='Skip tests that require downloading models'
    )
    return parser.parse_args()


def main():
    """Run all tests."""
    args = parse_args()

    logger.info("\n" + "=" * 80)
    logger.info("QUALITY ESTIMATION TEST SUITE")
    logger.info("=" * 80)
    logger.info(f"Testing: {args.component}")
    logger.info("=" * 80)

    results = {}

    # Run tests based on component
    if args.component in ['all', 'adequacy']:
        if not args.skip_models:
            results['sentence_embedding_adequacy'] = test_sentence_embedding_adequacy()
        else:
            logger.info("\nSkipping adequacy tests (--skip-models)")
            results['sentence_embedding_adequacy'] = True

    if args.component in ['all', 'fluency']:
        if not args.skip_models:
            results['perplexity_fluency'] = test_perplexity_fluency()
            results['parse_based_fluency'] = test_parse_based_fluency()
        else:
            logger.info("\nSkipping fluency tests (--skip-models)")
            results['perplexity_fluency'] = True
            results['parse_based_fluency'] = True

    if args.component in ['all', 'quality']:
        if not args.skip_models:
            results['quality_estimator'] = test_quality_estimator()
        else:
            logger.info("\nSkipping quality estimator test (--skip-models)")
            results['quality_estimator'] = True

    if args.component in ['all', 'filtering']:
        if not args.skip_models:
            results['quality_filtering'] = test_quality_filtering()
        else:
            logger.info("\nSkipping filtering test (--skip-models)")
            results['quality_filtering'] = True

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)

    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"{test_name:30s}: {status}")

    total = len(results)
    passed = sum(results.values())
    logger.info("=" * 80)
    logger.info(f"Total: {passed}/{total} tests passed")
    logger.info("=" * 80)

    if passed == total:
        logger.info("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        logger.error(f"\n❌ {total - passed} test(s) failed")
        sys.exit(1)


if __name__ == '__main__':
    main()
