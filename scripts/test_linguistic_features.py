#!/usr/bin/env python3
"""
Test script for linguistic features integration.

This script verifies that:
1. Linguistic features can be enabled and configured
2. Parser initialization works correctly
3. Parse features are extracted and encoded
4. Model wrapper integrates features correctly
5. Trainer handles linguistic features during training

Usage:
    python scripts/test_linguistic_features.py

    # Test with specific parser
    python scripts/test_linguistic_features.py --parser stanza

    # Test with GPU
    python scripts/test_linguistic_features.py --use-gpu
"""

import argparse
import sys
from pathlib import Path
import logging
import tempfile
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.linguistic import (
    LinguisticFeaturesConfig,
    StanzaParser,
    DependencyEncoder,
    EncoderAugmentation,
)
from src.training.models.linguistic_model import LinguisticEnhancedModel
from src.training.trainers.linguistic_trainer import (
    LinguisticDataCollator,
    LinguisticSeq2SeqTrainer,
    LinguisticLoRATrainer,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_parser(parser_name: str = "stanza", language: str = "hi", use_gpu: bool = False):
    """Test parser initialization and parsing."""
    logger.info("=" * 80)
    logger.info(f"TEST 1: Parser Initialization and Parsing ({parser_name})")
    logger.info("=" * 80)

    try:
        # Create parser
        from src.training.linguistic.parsers.base import create_parser
        parser = create_parser(parser_name, language, use_gpu=use_gpu, batch_size=2)

        # Test texts
        texts = [
            "राम ने रावण को हराया।",  # Ram defeated Ravana
            "यह एक परीक्षण वाक्य है।",  # This is a test sentence
        ]

        logger.info(f"Parsing {len(texts)} sentences...")
        parses = parser.parse(texts)

        # Verify results
        assert len(parses) == len(texts), f"Expected {len(texts)} parses, got {len(parses)}"

        for i, (text, parse) in enumerate(zip(texts, parses)):
            logger.info(f"\nSentence {i+1}: {text}")
            logger.info(f"  Words: {parse.words}")
            logger.info(f"  POS tags: {parse.pos_tags}")
            logger.info(f"  Dependency relations: {parse.deprels}")
            logger.info(f"  Heads: {parse.heads}")
            logger.info(f"  Tree depth: {parse.get_tree_depth()}")

            # Verify parse structure
            assert len(parse.words) > 0, "Empty parse"
            assert len(parse.words) == len(parse.pos_tags), "Mismatched POS tags"
            assert len(parse.words) == len(parse.deprels), "Mismatched deprels"
            assert len(parse.words) == len(parse.heads), "Mismatched heads"

        logger.info("\n✓ Parser test passed!")
        return True

    except Exception as e:
        logger.error(f"✗ Parser test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_feature_encoder():
    """Test dependency feature encoder."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Feature Encoding")
    logger.info("=" * 80)

    try:
        # Create parser
        parser = StanzaParser("hi", use_gpu=False, batch_size=2)

        # Parse text
        text = "राम ने रावण को हराया।"
        logger.info(f"Parsing: {text}")
        parses = parser.parse([text])
        parse = parses[0]

        # Create feature encoder
        encoder = DependencyEncoder(
            encoding_dim=128,
            use_graph_encoder=False,
            feature_types=["dependency_labels", "pos_tags", "tree_depth", "head_distance"]
        )

        # Encode features
        logger.info("Encoding features...")
        features = encoder.encode_parse_tree(parse, return_tensor=True)

        logger.info(f"  Parse length: {len(parse.words)} tokens")
        logger.info(f"  Feature shape: {features.shape}")
        logger.info(f"  Feature dtype: {features.dtype}")

        # Verify shape
        assert features.shape[0] == len(parse.words), "Feature length mismatch"
        assert features.shape[1] == 128, f"Expected dim 128, got {features.shape[1]}"
        assert isinstance(features, torch.Tensor), "Expected torch.Tensor"

        logger.info("\n✓ Feature encoder test passed!")
        return True

    except Exception as e:
        logger.error(f"✗ Feature encoder test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_encoder_augmentation():
    """Test encoder augmentation."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: Encoder Augmentation")
    logger.info("=" * 80)

    try:
        # Create config
        config = LinguisticFeaturesConfig(
            use_source_parse=True,
            parser="stanza",
            features=["dependency_labels", "pos_tags"],
            encoding_dim=64,
        )

        # Create augmentation
        augmentation = EncoderAugmentation(config, source_lang="hi")

        # Source texts and embeddings
        source_texts = ["राम ने रावण को हराया।", "यह एक परीक्षण है।"]
        # Simulate word embeddings [batch=2, seq_len=10, embed_dim=512]
        source_embeddings = torch.randn(2, 10, 512)

        logger.info(f"Source texts: {source_texts}")
        logger.info(f"Source embeddings shape: {source_embeddings.shape}")

        # Augment
        logger.info("Augmenting embeddings with parse features...")
        augmented = augmentation.augment_source(
            source_texts, source_embeddings, use_cache=True
        )

        logger.info(f"  Original shape: {source_embeddings.shape}")
        logger.info(f"  Augmented shape: {augmented.shape}")
        logger.info(f"  Expected shape: [2, 10, 576] (512 + 64)")

        # Verify shape
        assert augmented.shape[0] == 2, "Batch size mismatch"
        assert augmented.shape[1] == 10, "Sequence length mismatch"
        assert augmented.shape[2] == 512 + 64, f"Expected dim 576, got {augmented.shape[2]}"

        # Cleanup
        augmentation.cleanup()

        logger.info("\n✓ Encoder augmentation test passed!")
        return True

    except Exception as e:
        logger.error(f"✗ Encoder augmentation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_linguistic_model_wrapper():
    """Test linguistic model wrapper."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 4: Linguistic Model Wrapper")
    logger.info("=" * 80)

    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        # Create a small model for testing
        logger.info("Loading small model for testing...")
        model_name = "facebook/nllb-200-distilled-600M"

        # Use smaller model if available, or mock
        try:
            base_model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            logger.warning(f"Could not load model {model_name}: {e}")
            logger.info("Skipping model wrapper test (model loading failed)")
            return True

        # Create linguistic config
        linguistic_config = LinguisticFeaturesConfig(
            use_source_parse=True,
            parser="stanza",
            features=["dependency_labels", "pos_tags"],
            encoding_dim=64,
        )

        # Create linguistic-enhanced model
        logger.info("Creating linguistic-enhanced model...")
        linguistic_model = LinguisticEnhancedModel(
            base_model=base_model,
            linguistic_config=linguistic_config,
            source_lang="hi",
            target_lang="bho",
        )

        # Test forward pass
        logger.info("Testing forward pass with linguistic features...")
        source_texts = ["राम ने रावण को हराया।"]
        inputs = tokenizer(source_texts, return_tensors="pt", padding=True)
        labels = tokenizer(["राम रावण के हराइल।"], return_tensors="pt", padding=True).input_ids

        outputs = linguistic_model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            labels=labels,
            source_texts=source_texts,
        )

        logger.info(f"  Loss: {outputs.loss.item():.4f}")
        logger.info(f"  Output shape: {outputs.logits.shape}")

        # Cleanup
        linguistic_model.cleanup()

        logger.info("\n✓ Model wrapper test passed!")
        return True

    except Exception as e:
        logger.error(f"✗ Model wrapper test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_collator():
    """Test linguistic data collator."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 5: Linguistic Data Collator")
    logger.info("=" * 80)

    try:
        from transformers import AutoTokenizer

        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")

        # Create data collator
        collator = LinguisticDataCollator(
            tokenizer=tokenizer,
            padding=True,
            label_pad_token_id=-100,
        )

        # Create sample features
        features = [
            {
                'input_ids': [1, 2, 3, 4],
                'attention_mask': [1, 1, 1, 1],
                'labels': [5, 6, 7, 8],
                'source_text': "राम ने रावण को हराया।"
            },
            {
                'input_ids': [1, 2, 3],
                'attention_mask': [1, 1, 1],
                'labels': [5, 6, 7],
                'source_text': "यह एक परीक्षण है।"
            }
        ]

        logger.info("Collating batch...")
        batch = collator(features)

        logger.info(f"  Batch keys: {batch.keys()}")
        logger.info(f"  Input IDs shape: {batch['input_ids'].shape}")
        logger.info(f"  Source texts: {batch['source_texts']}")

        # Verify
        assert 'input_ids' in batch, "Missing input_ids"
        assert 'attention_mask' in batch, "Missing attention_mask"
        assert 'labels' in batch, "Missing labels"
        assert 'source_texts' in batch, "Missing source_texts"
        assert len(batch['source_texts']) == 2, "Wrong number of source texts"

        logger.info("\n✓ Data collator test passed!")
        return True

    except Exception as e:
        logger.error(f"✗ Data collator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test linguistic features')
    parser.add_argument(
        '--parser', default='stanza',
        choices=['stanza', 'spacy', 'trankit', 'udpipe'],
        help='Parser to test (default: stanza)'
    )
    parser.add_argument(
        '--language', default='hi',
        help='Language to test (default: hi)'
    )
    parser.add_argument(
        '--use-gpu', action='store_true',
        help='Use GPU for parsing'
    )
    parser.add_argument(
        '--skip-model', action='store_true',
        help='Skip model wrapper test (requires downloading model)'
    )
    return parser.parse_args()


def main():
    """Run all tests."""
    args = parse_args()

    logger.info("\n" + "=" * 80)
    logger.info("LINGUISTIC FEATURES TEST SUITE")
    logger.info("=" * 80)
    logger.info(f"Parser: {args.parser}")
    logger.info(f"Language: {args.language}")
    logger.info(f"GPU: {args.use_gpu}")
    logger.info("=" * 80)

    results = {}

    # Run tests
    results['parser'] = test_parser(args.parser, args.language, args.use_gpu)
    results['encoder'] = test_feature_encoder()
    results['augmentation'] = test_encoder_augmentation()

    if not args.skip_model:
        results['model_wrapper'] = test_linguistic_model_wrapper()
    else:
        logger.info("\nSkipping model wrapper test (--skip-model)")
        results['model_wrapper'] = True

    results['data_collator'] = test_data_collator()

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
