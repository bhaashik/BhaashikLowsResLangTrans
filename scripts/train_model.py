#!/usr/bin/env python3
"""
CLI for training machine translation models with LoRA fine-tuning.

Supports:
- All HuggingFace seq2seq models (NLLB, mBART, mT5, IndicTrans2)
- Causal LMs (Llama, Mistral) with instruction formatting
- LoRA parameter-efficient fine-tuning
- Multiple data formats (TSV, CSV, JSON, JSONL, HuggingFace)
- Configurable training hyperparameters
- Automatic evaluation and checkpointing
- Linguistic features integration (dependency parsing)

Usage:
    # Train NLLB with LoRA
    python scripts/train_model.py \\
        --model nllb-600m \\
        --source-lang hi \\
        --target-lang bho \\
        --train-data data/hi_bho_train.tsv \\
        --output models/nllb-bhojpuri

    # Train with linguistic features (dependency parsing)
    python scripts/train_model.py \\
        --model nllb-600m \\
        --source-lang hi \\
        --target-lang bho \\
        --train-data data/hi_bho_train.tsv \\
        --output models/nllb-bhojpuri-linguistic \\
        --use-linguistic-features \\
        --linguistic-parser stanza \\
        --linguistic-encoding-dim 128

    # Train with custom config
    python scripts/train_model.py \\
        --model nllb-1.3b \\
        --source-lang hi \\
        --target-lang mwr \\
        --train-data data/train.tsv \\
        --val-data data/val.tsv \\
        --config config/training/nllb_custom.yaml

    # Train causal LM
    python scripts/train_model.py \\
        --model llama3-8b \\
        --source-lang en \\
        --target-lang hi \\
        --train-data data/train.json \\
        --epochs 5 \\
        --batch-size 8 \\
        --lora-r 32
"""

import argparse
import sys
from pathlib import Path
import logging
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.config import (
    TrainingConfig,
    LoRAConfig,
    ModelConfig,
    DataConfig,
)
from src.training.trainers import LoRATrainer, CausalLMTrainer
from src.training.trainers.linguistic_trainer import create_trainer
from src.training.models import ModelFactory
from src.training.data import ParallelCorpusLoader
from src.training.linguistic import LinguisticFeaturesConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train MT models with LoRA fine-tuning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Model selection
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument(
        '--model', required=True,
        help='Model preset name (nllb-600m, mbart-large, mt5-base, llama3-8b, etc.)'
    )
    model_group.add_argument(
        '--source-lang', required=True,
        help='Source language code (e.g., hi, en)'
    )
    model_group.add_argument(
        '--target-lang', required=True,
        help='Target language code (e.g., bho, mwr)'
    )

    # Data configuration
    data_group = parser.add_argument_group('Data Configuration')
    data_group.add_argument(
        '--train-data', required=True,
        help='Path to training data (TSV/CSV/JSON/JSONL/HuggingFace dataset)'
    )
    data_group.add_argument(
        '--val-data',
        help='Path to validation data (optional, will split from train if not provided)'
    )
    data_group.add_argument(
        '--data-format',
        choices=['tsv', 'csv', 'json', 'jsonl', 'hf'],
        default='tsv',
        help='Data format (default: tsv)'
    )
    data_group.add_argument(
        '--source-column',
        default='source',
        help='Source column name (default: source)'
    )
    data_group.add_argument(
        '--target-column',
        default='target',
        help='Target column name (default: target)'
    )

    # Output configuration
    output_group = parser.add_argument_group('Output Configuration')
    output_group.add_argument(
        '--output', required=True,
        help='Output directory for trained model'
    )
    output_group.add_argument(
        '--save-steps', type=int, default=500,
        help='Save checkpoint every N steps (default: 500)'
    )
    output_group.add_argument(
        '--save-full-model', action='store_true',
        help='Save full model instead of adapter only'
    )

    # Training hyperparameters
    train_group = parser.add_argument_group('Training Hyperparameters')
    train_group.add_argument(
        '--epochs', type=int, default=10,
        help='Number of training epochs (default: 10)'
    )
    train_group.add_argument(
        '--batch-size', type=int, default=16,
        help='Per-device batch size (default: 16)'
    )
    train_group.add_argument(
        '--learning-rate', type=float, default=3e-4,
        help='Learning rate (default: 3e-4)'
    )
    train_group.add_argument(
        '--warmup-ratio', type=float, default=0.1,
        help='Warmup ratio (default: 0.1)'
    )
    train_group.add_argument(
        '--gradient-accumulation', type=int, default=4,
        help='Gradient accumulation steps (default: 4)'
    )

    # LoRA configuration
    lora_group = parser.add_argument_group('LoRA Configuration')
    lora_group.add_argument(
        '--lora-r', type=int, default=16,
        help='LoRA rank (default: 16)'
    )
    lora_group.add_argument(
        '--lora-alpha', type=int, default=32,
        help='LoRA alpha (default: 32)'
    )
    lora_group.add_argument(
        '--lora-dropout', type=float, default=0.1,
        help='LoRA dropout (default: 0.1)'
    )

    # Linguistic features configuration
    linguistic_group = parser.add_argument_group('Linguistic Features')
    linguistic_group.add_argument(
        '--use-linguistic-features', action='store_true',
        help='Enable linguistic features (dependency parsing)'
    )
    linguistic_group.add_argument(
        '--linguistic-parser',
        choices=['stanza', 'spacy', 'trankit', 'udpipe'],
        default='stanza',
        help='Parser to use for linguistic features (default: stanza)'
    )
    linguistic_group.add_argument(
        '--linguistic-features', nargs='+',
        default=['dependency_labels', 'pos_tags', 'tree_depth', 'head_distance'],
        help='Features to extract from parse (default: dependency_labels pos_tags tree_depth head_distance)'
    )
    linguistic_group.add_argument(
        '--linguistic-encoding-dim', type=int, default=128,
        help='Encoding dimension for linguistic features (default: 128)'
    )
    linguistic_group.add_argument(
        '--linguistic-use-source', action='store_true', default=True,
        help='Use source-side parse (default: True)'
    )
    linguistic_group.add_argument(
        '--linguistic-use-target', action='store_true',
        help='Use target-side parse'
    )
    linguistic_group.add_argument(
        '--linguistic-use-graph-encoder', action='store_true',
        help='Use GNN for parse encoding (experimental)'
    )

    # Advanced configuration
    advanced_group = parser.add_argument_group('Advanced Configuration')
    advanced_group.add_argument(
        '--config',
        help='Path to YAML config file (overrides command-line args)'
    )
    advanced_group.add_argument(
        '--device',
        choices=['cuda', 'cpu'],
        default='cuda',
        help='Device to use (default: cuda)'
    )
    advanced_group.add_argument(
        '--fp16', action='store_true',
        help='Use mixed precision training'
    )
    advanced_group.add_argument(
        '--resume-from-checkpoint',
        help='Resume training from checkpoint'
    )

    return parser.parse_args()


def load_config_from_yaml(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_configs_from_args(args):
    """Create configuration objects from command-line arguments."""

    # Load from YAML if provided
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        yaml_config = load_config_from_yaml(args.config)

        # Create configs from YAML
        training_config = TrainingConfig.from_dict(yaml_config.get('training', {}))
        lora_config = LoRAConfig.from_dict(yaml_config.get('lora', {}))
        model_config = ModelConfig.from_dict(yaml_config.get('model', {}))
        data_config = DataConfig.from_dict(yaml_config.get('data', {}))

        # Linguistic features config from YAML
        linguistic_config = None
        if 'linguistic' in yaml_config:
            linguistic_config = LinguisticFeaturesConfig.from_dict(yaml_config['linguistic'])

    else:
        # Create configs from command-line args

        # Model config
        model_config = ModelConfig.create_preset(
            args.model,
            source_lang=args.source_lang,
            target_lang=args.target_lang
        )
        model_config.device = args.device

        # LoRA config
        lora_config = LoRAConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )

        # Training config
        training_config = TrainingConfig(
            output_dir=args.output,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            warmup_ratio=args.warmup_ratio,
            gradient_accumulation_steps=args.gradient_accumulation,
            save_steps=args.save_steps,
            fp16=args.fp16 if args.device == 'cuda' else False,
        )

        # Data config
        data_config = DataConfig(
            train_file=args.train_data,
            val_file=args.val_data,
            format=args.data_format,
            source_column=args.source_column,
            target_column=args.target_column,
        )

        # Linguistic features config from command-line args
        linguistic_config = None
        if args.use_linguistic_features:
            linguistic_config = LinguisticFeaturesConfig(
                use_source_parse=args.linguistic_use_source,
                use_target_parse=args.linguistic_use_target,
                parser=args.linguistic_parser,
                features=args.linguistic_features,
                encoding_dim=args.linguistic_encoding_dim,
                use_graph_encoder=args.linguistic_use_graph_encoder,
            )

    return model_config, lora_config, training_config, data_config, linguistic_config


def main():
    """Main training function."""
    args = parse_args()

    logger.info("=" * 80)
    logger.info("MT Model Training with LoRA")
    logger.info("=" * 80)

    # Create configurations
    model_config, lora_config, training_config, data_config, linguistic_config = create_configs_from_args(args)

    logger.info(f"Model: {model_config.model_name_or_path}")
    logger.info(f"Languages: {args.source_lang} → {args.target_lang}")
    logger.info(f"Training data: {data_config.train_file}")
    logger.info(f"Output: {training_config.output_dir}")

    if linguistic_config:
        logger.info("\n" + "=" * 80)
        logger.info("Linguistic Features: ENABLED")
        logger.info("=" * 80)
        logger.info(f"  Parser: {linguistic_config.parser}")
        logger.info(f"  Features: {', '.join(linguistic_config.features)}")
        logger.info(f"  Encoding dim: {linguistic_config.encoding_dim}")
        logger.info(f"  Source parse: {linguistic_config.use_source_parse}")
        logger.info(f"  Target parse: {linguistic_config.use_target_parse}")
        logger.info(f"  Integration: {linguistic_config.integration_method}")
        logger.info("=" * 80)

    # Load data
    logger.info("\nLoading training data...")
    data_loader = ParallelCorpusLoader(
        train_file=data_config.train_file,
        val_file=data_config.val_file,
        format=data_config.format,
        source_column=data_config.source_column,
        target_column=data_config.target_column,
        test_size=data_config.test_size,
        val_size=data_config.val_size,
        max_length=data_config.max_length,
        min_length=data_config.min_length,
        filter_duplicates=data_config.filter_duplicates,
    )

    dataset_dict = data_loader.load()
    logger.info(f"Loaded {len(dataset_dict['train'])} training examples")
    if 'validation' in dataset_dict:
        logger.info(f"Loaded {len(dataset_dict['validation'])} validation examples")

    # Create model
    logger.info("\nCreating model...")
    model = ModelFactory.create_from_config(model_config)

    # Create trainer (applies LoRA and linguistic features internally)
    logger.info("\nInitializing trainer...")
    if model_config.model_type == "causal_lm":
        # For causal LMs, use CausalLMTrainer (no linguistic features support yet)
        model.prepare_for_training(lora_config, training_config)
        trainer = CausalLMTrainer(
            model=model,
            train_dataset=dataset_dict['train'],
            eval_dataset=dataset_dict.get('validation'),
            training_config=training_config,
        )
    else:
        # For seq2seq models, use create_trainer() which handles linguistic features
        trainer = create_trainer(
            model=model,
            training_config=training_config,
            train_dataset=dataset_dict['train'],
            eval_dataset=dataset_dict.get('validation'),
            lora_config=lora_config,
            linguistic_config=linguistic_config,
        )

    # Train
    logger.info("\n" + "=" * 80)
    logger.info("Starting training...")
    logger.info("=" * 80)

    metrics = trainer.train(
        resume_from_checkpoint=args.resume_from_checkpoint
    )

    logger.info("\n" + "=" * 80)
    logger.info("Training completed!")
    logger.info("=" * 80)
    logger.info(f"Final metrics: {metrics}")
    logger.info(f"Model saved to: {training_config.output_dir}")

    # Save full model if requested
    if args.save_full_model:
        logger.info("\nMerging and saving full model...")
        model.save(
            output_dir=str(Path(training_config.output_dir) / "merged"),
            save_full_model=True
        )
        logger.info(f"Full model saved to: {training_config.output_dir}/merged")


if __name__ == '__main__':
    main()
