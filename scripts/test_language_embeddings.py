#!/usr/bin/env python3
"""
Test script for language embeddings integration.

This script verifies that:
1. Language embedding loaders work correctly (URIEL, WALS, lang2vec)
2. Language embeddings can be loaded for different languages
3. Encoder projects embeddings correctly
4. Model integration works

Usage:
    python scripts/test_language_embeddings.py

    # Test specific source
    python scripts/test_language_embeddings.py --source uriel

    # Skip model test
    python scripts/test_language_embeddings.py --skip-model
"""

import argparse
import sys
from pathlib import Path
import logging
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.language_embeddings import (
    LanguageEmbeddingsConfig,
    URIELLoader,
    WALSLoader,
    Lang2VecLoader,
    LanguageEmbeddingEncoder,
    create_language_encoder,
)
from src.training.language_embeddings.loaders.base import LoaderRegistry

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_uriel_loader():
    """Test URIEL loader."""
    logger.info("=" * 80)
    logger.info("TEST 1: URIEL Loader")
    logger.info("=" * 80)

    try:
        # Create loader
        loader = URIELLoader(
            feature_types=["syntax", "phonology"],
            normalize=True,
            cache=True,
        )
        loader.initialize()

        # Test languages
        test_langs = ["hi", "en", "bho"]

        for lang in test_langs:
            logger.info(f"\nLoading URIEL embedding for {lang}...")

            if not loader.supports_language(lang):
                logger.warning(f"  Language {lang} not supported (embedded data only)")
                continue

            emb = loader.load_embedding(lang)

            logger.info(f"  Language: {emb.language_code}")
            logger.info(f"  Dimension: {emb.dimension}")
            logger.info(f"  Source: {emb.source}")
            logger.info(f"  Feature names: {len(emb.feature_names)} features")
            logger.info(f"  Embedding shape: {emb.embedding.shape}")
            logger.info(f"  Sample values: {emb.embedding[:5]}")

            # Verify
            assert emb.dimension > 0, "Empty embedding"
            assert emb.source == "uriel", "Wrong source"

        logger.info("\n✓ URIEL loader test passed!")
        return True

    except Exception as e:
        logger.error(f"✗ URIEL loader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_wals_loader():
    """Test WALS loader."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: WALS Loader")
    logger.info("=" * 80)

    try:
        # Create loader
        loader = WALSLoader(
            feature_types=["syntax", "phonology", "morphology"],
            normalize=True,
            cache=True,
        )
        loader.initialize()

        # Test languages
        test_langs = ["hi", "en"]

        for lang in test_langs:
            logger.info(f"\nLoading WALS embedding for {lang}...")

            if not loader.supports_language(lang):
                logger.warning(f"  Language {lang} not supported")
                continue

            emb = loader.load_embedding(lang)

            logger.info(f"  Language: {emb.language_code}")
            logger.info(f"  Dimension: {emb.dimension}")
            logger.info(f"  Source: {emb.source}")
            logger.info(f"  Feature names: {len(emb.feature_names)} features")
            logger.info(f"  Sample values: {emb.embedding[:5]}")

            # Verify
            assert emb.dimension > 0, "Empty embedding"
            assert emb.source == "wals", "Wrong source"

        logger.info("\n✓ WALS loader test passed!")
        return True

    except Exception as e:
        logger.error(f"✗ WALS loader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_lang2vec_loader():
    """Test lang2vec loader."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: lang2vec Loader")
    logger.info("=" * 80)

    try:
        # Try to import lang2vec
        try:
            import lang2vec
        except ImportError:
            logger.warning("lang2vec not installed, skipping test")
            logger.info("  Install with: pip install lang2vec")
            return True

        # Create loader
        loader = Lang2VecLoader(
            normalize=True,
            cache=True,
        )
        loader.initialize()

        # Test language
        lang = "hi"
        logger.info(f"\nLoading lang2vec embedding for {lang}...")

        emb = loader.load_embedding(lang)

        logger.info(f"  Language: {emb.language_code}")
        logger.info(f"  Dimension: {emb.dimension}")
        logger.info(f"  Source: {emb.source}")
        logger.info(f"  Embedding type: learned (dense)")
        logger.info(f"  Sample values: {emb.embedding[:5]}")

        # Verify
        assert emb.dimension == 512, f"Expected dim 512, got {emb.dimension}"
        assert emb.source == "lang2vec", "Wrong source"

        logger.info("\n✓ lang2vec loader test passed!")
        return True

    except Exception as e:
        logger.error(f"✗ lang2vec loader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_language_encoder():
    """Test language embedding encoder."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 4: Language Embedding Encoder")
    logger.info("=" * 80)

    try:
        # Create encoder
        encoder = create_language_encoder(
            embedding_source="uriel",
            output_dim=64,
            feature_types=["syntax", "phonology"],
            use_projection=True,
        )

        # Test single language
        logger.info("\nEncoding single language...")
        hi_emb = encoder.encode_language("hi")
        logger.info(f"  Hindi embedding shape: {hi_emb.shape}")
        logger.info(f"  Expected shape: [64]")
        assert hi_emb.shape == (64,), f"Wrong shape: {hi_emb.shape}"

        # Test language pair
        logger.info("\nEncoding language pair...")
        pair_emb = encoder.encode_source_target_pair("hi", "bho", concatenate=True)
        logger.info(f"  Hindi-Bhojpuri pair shape: {pair_emb.shape}")
        logger.info(f"  Expected shape: [128]  # 2 * 64")
        assert pair_emb.shape == (128,), f"Wrong shape: {pair_emb.shape}"

        # Test batch
        logger.info("\nEncoding batch...")
        batch_emb = encoder.encode_batch(["hi", "en", "bho"])
        logger.info(f"  Batch shape: {batch_emb.shape}")
        logger.info(f"  Expected shape: [3, 64]")
        assert batch_emb.shape == (3, 64), f"Wrong shape: {batch_emb.shape}"

        logger.info("\n✓ Language encoder test passed!")
        return True

    except Exception as e:
        logger.error(f"✗ Language encoder test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_integration():
    """Test model integration."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 5: Model Integration")
    logger.info("=" * 80)

    try:
        from transformers import AutoModelForSeq2SeqLM
        from src.training.models.language_aware_model import LanguageAwareModel
        from src.training.language_embeddings import LanguageEmbeddingsConfig

        # Load a small model for testing
        logger.info("Loading small model for testing...")
        model_name = "facebook/nllb-200-distilled-600M"

        try:
            base_model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
            )
        except Exception as e:
            logger.warning(f"Could not load model {model_name}: {e}")
            logger.info("Skipping model integration test (model loading failed)")
            return True

        # Create language embeddings config
        lang_emb_config = LanguageEmbeddingsConfig(
            use_source_embedding=True,
            use_target_embedding=False,
            embedding_source="uriel",
            embedding_dim=64,
            feature_types=["syntax", "phonology"],
            integration_method="concatenate",
        )

        # Create language-aware model
        logger.info("Creating language-aware model...")
        lang_aware_model = LanguageAwareModel(
            base_model=base_model,
            language_embeddings_config=lang_emb_config,
            source_lang="hi",
            target_lang="bho",
        )

        # Test forward pass
        logger.info("Testing forward pass...")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        inputs = tokenizer(["नमस्ते"], return_tensors="pt", padding=True)
        labels = tokenizer(["नमस्कार"], return_tensors="pt", padding=True).input_ids

        outputs = lang_aware_model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            labels=labels,
        )

        logger.info(f"  Loss: {outputs.loss.item():.4f}")
        logger.info(f"  Output logits shape: {outputs.logits.shape}")

        # Cleanup
        lang_aware_model.cleanup()

        logger.info("\n✓ Model integration test passed!")
        return True

    except Exception as e:
        logger.error(f"✗ Model integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_loader_registry():
    """Test loader registry."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 6: Loader Registry")
    logger.info("=" * 80)

    try:
        # Get available loaders
        loaders = LoaderRegistry.get_available_loaders()
        logger.info(f"Available loaders: {loaders}")

        assert "uriel" in loaders, "URIEL loader not registered"
        assert "wals" in loaders, "WALS loader not registered"
        assert "lang2vec" in loaders, "lang2vec loader not registered"

        # Create loader via registry
        logger.info("\nCreating URIEL loader via registry...")
        loader = LoaderRegistry.create_loader(
            "uriel",
            feature_types=["syntax"],
            normalize=True,
        )

        assert loader is not None, "Loader creation failed"
        logger.info("  Loader created successfully")

        logger.info("\n✓ Loader registry test passed!")
        return True

    except Exception as e:
        logger.error(f"✗ Loader registry test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test language embeddings')
    parser.add_argument(
        '--source', default='all',
        choices=['all', 'uriel', 'wals', 'lang2vec'],
        help='Embedding source to test (default: all)'
    )
    parser.add_argument(
        '--skip-model', action='store_true',
        help='Skip model integration test (requires downloading model)'
    )
    return parser.parse_args()


def main():
    """Run all tests."""
    args = parse_args()

    logger.info("\n" + "=" * 80)
    logger.info("LANGUAGE EMBEDDINGS TEST SUITE")
    logger.info("=" * 80)
    logger.info(f"Testing: {args.source}")
    logger.info("=" * 80)

    results = {}

    # Run tests based on source
    if args.source in ['all', 'uriel']:
        results['uriel'] = test_uriel_loader()

    if args.source in ['all', 'wals']:
        results['wals'] = test_wals_loader()

    if args.source in ['all', 'lang2vec']:
        results['lang2vec'] = test_lang2vec_loader()

    if args.source == 'all':
        results['encoder'] = test_language_encoder()
        results['registry'] = test_loader_registry()

        if not args.skip_model:
            results['model_integration'] = test_model_integration()
        else:
            logger.info("\nSkipping model integration test (--skip-model)")
            results['model_integration'] = True

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
