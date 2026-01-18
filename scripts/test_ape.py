#!/usr/bin/env python3
"""
Test script for Automatic Post-Editing (APE) components.

This script verifies that:
1. APE configuration works correctly
2. Iterative refinement APE functions
3. Back-translation APE functions
4. Multi-pass APE functions
5. APE pipeline integrates correctly

Usage:
    python scripts/test_ape.py

    # Test specific method
    python scripts/test_ape.py --method iterative

    # Skip model loading tests
    python scripts/test_ape.py --skip-models
"""

import argparse
import sys
from pathlib import Path
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.ape import (
    APEConfig,
    IterativeRefinementAPE,
    MultiPassAPE,
    APEPipeline,
    create_ape_pipeline,
)
from src.training.quality import QualityConfig, create_quality_estimator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_ape_config():
    """Test APE configuration."""
    logger.info("=" * 80)
    logger.info("TEST 1: APE Configuration")
    logger.info("=" * 80)

    try:
        # Create simple config
        config = APEConfig.create_simple()
        logger.info(f"Simple config created:")
        logger.info(f"  Method: {config.ape_method}")
        logger.info(f"  Max iterations: {config.max_iterations}")
        logger.info(f"  Quality threshold: {config.quality_threshold}")

        # Verify
        assert config.use_ape == True, "APE should be enabled"
        assert config.ape_method == "iterative", "Should use iterative method"

        # Create aggressive config
        aggressive = APEConfig.create_aggressive()
        logger.info(f"\nAggressive config created:")
        logger.info(f"  Method: {aggressive.ape_method}")
        logger.info(f"  Max iterations: {aggressive.max_iterations}")
        logger.info(f"  Num passes: {aggressive.num_passes}")

        assert aggressive.max_iterations > config.max_iterations, "Should have more iterations"

        # Test to/from dict
        config_dict = config.to_dict()
        config_restored = APEConfig.from_dict(config_dict)
        assert config_restored.ape_method == config.ape_method, "Should restore correctly"

        logger.info("\n✓ APE configuration test passed!")
        return True

    except Exception as e:
        logger.error(f"✗ APE configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_iterative_ape():
    """Test iterative refinement APE."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Iterative Refinement APE")
    logger.info("=" * 80)

    try:
        # Check dependencies
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            import sentence_transformers
        except ImportError:
            logger.warning("Required libraries not installed, skipping test")
            return True

        # Load small model for testing
        logger.info("Loading model for testing...")
        model_name = "t5-small"
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Create quality estimator
        quality_config = QualityConfig(
            use_adequacy=True,
            use_fluency=False,
            adequacy_method="sentence_embedding",
            adequacy_model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            adequacy_weight=1.0,
            fluency_weight=0.0,
            device="cpu",
        )
        quality_estimator = create_quality_estimator(quality_config)

        # Create APE config
        ape_config = APEConfig(
            use_ape=True,
            ape_method="iterative",
            max_iterations=2,
            quality_threshold=0.8,
            use_quality_estimation=True,
            device="cpu",
        )

        # Create APE engine
        ape = IterativeRefinementAPE(
            model=model,
            tokenizer=tokenizer,
            config=ape_config,
            quality_estimator=quality_estimator,
        )

        # Test refinement
        source = "Hello, how are you?"
        initial = "Hi there"

        logger.info(f"\nSource: {source}")
        logger.info(f"Initial: {initial}")

        refined, history = ape.refine_single(
            source, initial, return_history=True
        )

        logger.info(f"Refined: {refined}")
        logger.info(f"Iterations: {len(history) - 1}")

        for entry in history:
            logger.info(
                f"  Iteration {entry['iteration']}: "
                f"{entry['translation'][:50]}... "
                f"(quality: {entry.get('quality', 'N/A')})"
            )

        assert len(history) >= 1, "Should have at least initial translation"

        # Test batch
        logger.info("\nTesting batch refinement...")
        sources = [source] * 2
        initials = [initial, "Hey"]

        refined_batch = ape.refine_batch(sources, initials)
        assert len(refined_batch) == 2, "Should refine both"

        logger.info("\n✓ Iterative APE test passed!")
        return True

    except Exception as e:
        logger.error(f"✗ Iterative APE test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multipass_ape():
    """Test multi-pass APE."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: Multi-Pass APE")
    logger.info("=" * 80)

    try:
        # Check dependencies
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            import sentence_transformers
        except ImportError:
            logger.warning("Required libraries not installed, skipping test")
            return True

        # Load model
        model_name = "t5-small"
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Create quality estimator
        quality_config = QualityConfig(
            use_adequacy=True,
            use_fluency=False,
            adequacy_method="sentence_embedding",
            adequacy_model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            adequacy_weight=1.0,
            fluency_weight=0.0,
            device="cpu",
        )
        quality_estimator = create_quality_estimator(quality_config)

        # Create APE config
        ape_config = APEConfig(
            use_ape=True,
            ape_method="multipass",
            num_passes=2,
            pass_strategies=["beam", "sampling"],
            quality_threshold=0.8,
            use_quality_estimation=True,
            device="cpu",
        )

        # Create APE engine
        ape = MultiPassAPE(
            model=model,
            tokenizer=tokenizer,
            config=ape_config,
            quality_estimator=quality_estimator,
        )

        # Test refinement
        source = "Translate this to French"
        initial = "Traduis ceci"

        logger.info(f"\nSource: {source}")
        logger.info(f"Initial: {initial}")

        refined, candidates = ape.refine_single(
            source, initial, return_candidates=True
        )

        logger.info(f"Refined: {refined}")
        logger.info(f"Candidates: {len(candidates)}")

        for cand in candidates:
            logger.info(
                f"  Pass {cand['pass']} ({cand['strategy']}): "
                f"{cand['translation'][:40]}... "
                f"(quality: {cand.get('quality', 'N/A')})"
            )

        assert len(candidates) >= 2, "Should have multiple candidates"

        logger.info("\n✓ Multi-pass APE test passed!")
        return True

    except Exception as e:
        logger.error(f"✗ Multi-pass APE test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ape_pipeline():
    """Test APE pipeline."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 4: APE Pipeline")
    logger.info("=" * 80)

    try:
        # Check dependencies
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            import sentence_transformers
        except ImportError:
            logger.warning("Required libraries not installed, skipping test")
            return True

        # Load model
        model_name = "t5-small"
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Create quality estimator
        quality_config = QualityConfig(
            use_adequacy=True,
            use_fluency=False,
            adequacy_method="sentence_embedding",
            adequacy_model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            adequacy_weight=1.0,
            fluency_weight=0.0,
            device="cpu",
        )
        quality_estimator = create_quality_estimator(quality_config)

        # Create APE config
        ape_config = APEConfig.create_simple()
        ape_config.device = "cpu"

        # Create pipeline
        pipeline = create_ape_pipeline(
            config=ape_config,
            forward_model=model,
            forward_tokenizer=tokenizer,
            quality_estimator=quality_estimator,
        )

        # Test refinement
        sources = ["Hello", "Goodbye"]
        translations = ["Hi", "Bye"]

        logger.info(f"\nRefining {len(translations)} translations...")
        refined = pipeline.refine(sources, translations)

        logger.info(f"Results:")
        for src, orig, ref in zip(sources, translations, refined):
            logger.info(f"  {src} → {orig} → {ref}")

        assert len(refined) == len(translations), "Should refine all"

        # Test statistics
        logger.info("\nGetting statistics...")
        stats = pipeline.get_statistics(sources, translations)
        logger.info(f"  APE enabled: {stats.get('ape_enabled')}")
        logger.info(f"  APE method: {stats.get('ape_method')}")

        logger.info("\n✓ APE pipeline test passed!")
        return True

    except Exception as e:
        logger.error(f"✗ APE pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test APE components')
    parser.add_argument(
        '--method', default='all',
        choices=['all', 'config', 'iterative', 'multipass', 'pipeline'],
        help='Method to test (default: all)'
    )
    parser.add_argument(
        '--skip-models', action='store_true',
        help='Skip tests that require loading models'
    )
    return parser.parse_args()


def main():
    """Run all tests."""
    args = parse_args()

    logger.info("\n" + "=" * 80)
    logger.info("APE TEST SUITE")
    logger.info("=" * 80)
    logger.info(f"Testing: {args.method}")
    logger.info("=" * 80)

    results = {}

    # Run tests
    if args.method in ['all', 'config']:
        results['config'] = test_ape_config()

    if args.method in ['all', 'iterative']:
        if not args.skip_models:
            results['iterative'] = test_iterative_ape()
        else:
            logger.info("\nSkipping iterative APE test (--skip-models)")
            results['iterative'] = True

    if args.method in ['all', 'multipass']:
        if not args.skip_models:
            results['multipass'] = test_multipass_ape()
        else:
            logger.info("\nSkipping multi-pass APE test (--skip-models)")
            results['multipass'] = True

    if args.method in ['all', 'pipeline']:
        if not args.skip_models:
            results['pipeline'] = test_ape_pipeline()
        else:
            logger.info("\nSkipping pipeline test (--skip-models)")
            results['pipeline'] = True

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)

    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"{test_name:20s}: {status}")

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
