#!/usr/bin/env python3
"""
Basic tests for configuration management.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import Config


def test_config_initialization():
    """Test that config initializes correctly."""
    config = Config()
    assert config.base_dir is not None
    print("✓ Config initialization successful")


def test_config_get():
    """Test config.get() method."""
    config = Config()

    # Test nested key access
    value = config.get('languages.indictrans2_supported')
    assert isinstance(value, list)
    assert 'hi' in value  # Hindi should be supported
    print("✓ Config get() method works")


def test_supported_languages():
    """Test getting supported languages."""
    config = Config()

    supported = config.get_supported_languages()
    assert isinstance(supported, list)
    assert len(supported) > 0
    print(f"✓ Found {len(supported)} supported languages")


def test_unsupported_languages():
    """Test getting unsupported languages."""
    config = Config()

    unsupported = config.get_unsupported_languages()
    assert isinstance(unsupported, dict)
    assert 'bhojpuri' in unsupported or len(unsupported) == 0
    print(f"✓ Found {len(unsupported)} unsupported language groups")


def test_model_paths():
    """Test getting model paths."""
    config = Config()

    # Test IndicTrans2 path
    model_path = config.get_model_path('indictrans2', 'default')
    assert model_path is not None
    assert 'indictrans2' in model_path
    print(f"✓ Model path retrieval works: {model_path}")


def test_translation_strategy():
    """Test getting translation strategy."""
    config = Config()

    # Test supported language strategy
    strategy_hi = config.get_translation_strategy('hi')
    assert strategy_hi['method'] == 'indictrans2_direct'
    assert strategy_hi['cost'] == 0
    print("✓ Translation strategy for supported language works")

    # Test unsupported language strategy
    strategy_bho = config.get_translation_strategy('bho')
    assert 'method' in strategy_bho
    print("✓ Translation strategy for unsupported language works")


def test_api_costs():
    """Test getting API costs."""
    config = Config()

    costs = config.get_api_costs('anthropic', 'haiku_3_5')
    assert 'input_cost_per_1m_tokens' in costs
    assert 'output_cost_per_1m_tokens' in costs
    print(f"✓ API costs retrieved: {costs}")


def test_paths():
    """Test getting configured paths."""
    config = Config()

    paths = config.get_paths()
    assert isinstance(paths, dict)
    assert 'datasets' in paths
    assert 'models' in paths
    assert 'logs' in paths
    print(f"✓ Found {len(paths)} configured paths")


def run_all_tests():
    """Run all tests."""
    print("="*70)
    print("RUNNING CONFIGURATION TESTS")
    print("="*70)

    tests = [
        test_config_initialization,
        test_config_get,
        test_supported_languages,
        test_unsupported_languages,
        test_model_paths,
        test_translation_strategy,
        test_api_costs,
        test_paths
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            print(f"\n{test.__name__}:")
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {str(e)}")
            failed += 1

    print("\n" + "="*70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*70)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
