"""
Trainer with linguistic features support.

Extends LoRATrainer to support training with dependency parse features.
"""

from typing import Optional, Dict, Any, List
from pathlib import Path
from datasets import Dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from dataclasses import dataclass
import torch
import logging

from src.training.trainers.base import AbstractTrainer
from src.training.trainers.metrics import compute_metrics
from src.training.config import TrainingConfig, LoRAConfig
from src.training.models.base import AbstractTrainableMTModel
from src.training.linguistic import LinguisticFeaturesConfig

logger = logging.getLogger(__name__)


@dataclass
class LinguisticDataCollator:
    """
    Data collator that includes source texts for linguistic features.

    Extends DataCollatorForSeq2Seq to also collect source texts.
    """

    tokenizer: Any
    model: Optional[Any] = None
    padding: bool = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate batch and extract source texts.

        Args:
            features: List of feature dicts from dataset

        Returns:
            Batch dict with source_texts included
        """
        # Extract source texts if available
        source_texts = None
        if features and 'source_text' in features[0]:
            source_texts = [f['source_text'] for f in features]

            # Remove source_text from features (not needed for collation)
            features = [
                {k: v for k, v in f.items() if k != 'source_text'}
                for f in features
            ]

        # Use standard data collator for the rest
        from transformers import DataCollatorForSeq2Seq as StandardCollator

        standard_collator = StandardCollator(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            label_pad_token_id=self.label_pad_token_id,
        )

        batch = standard_collator(features)

        # Add source texts to batch
        if source_texts:
            batch['source_texts'] = source_texts

        return batch


class LinguisticSeq2SeqTrainer(Seq2SeqTrainer):
    """
    Seq2Seq trainer that passes source_texts to model.

    This allows the model to use linguistic features during training.
    """

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute loss with source_texts passed to model.

        Args:
            model: Model to train
            inputs: Batch inputs
            return_outputs: Whether to return model outputs

        Returns:
            Loss (and optionally outputs)
        """
        # Extract source_texts if present
        source_texts = inputs.pop('source_texts', None)

        # Add source_texts to inputs if model supports it
        if source_texts is not None and hasattr(model, 'augmentation'):
            inputs['source_texts'] = source_texts

        # Standard loss computation
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        outputs = model(**inputs)

        if labels is not None:
            if isinstance(outputs, dict):
                loss = self.label_smoother(outputs, labels)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict):
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            else:
                loss = outputs[0]

        return (loss, outputs) if return_outputs else loss


class LinguisticLoRATrainer(AbstractTrainer):
    """
    Trainer for fine-tuning seq2seq models with LoRA and linguistic features.

    Extends LoRATrainer to support dependency parse features.
    """

    def __init__(
        self,
        model: AbstractTrainableMTModel,
        training_config: TrainingConfig,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        lora_config: Optional[LoRAConfig] = None,
        linguistic_config: Optional[LinguisticFeaturesConfig] = None,
    ):
        """
        Initialize linguistic LoRA trainer.

        Args:
            model: Trainable MT model
            training_config: Training configuration
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            lora_config: LoRA configuration (None for full fine-tuning)
            linguistic_config: Linguistic features configuration
        """
        super().__init__(model, training_config)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.lora_config = lora_config
        self.linguistic_config = linguistic_config

        # Prepare datasets (add source_text column if needed)
        if self.linguistic_config:
            self._prepare_datasets_for_linguistic()

        # Prepare model for training
        self._prepare_model()

        # Setup trainer
        self._setup_trainer()

    def _prepare_datasets_for_linguistic(self):
        """Ensure datasets have source_text column for linguistic features."""
        # Check if source_text column exists
        if 'source_text' not in self.train_dataset.column_names:
            # Try to get from 'source' or 'src' column
            if 'source' in self.train_dataset.column_names:
                self.train_dataset = self.train_dataset.map(
                    lambda x: {'source_text': x['source']},
                    desc="Adding source_text column"
                )
            elif 'src' in self.train_dataset.column_names:
                self.train_dataset = self.train_dataset.map(
                    lambda x: {'source_text': x['src']},
                    desc="Adding source_text column"
                )
            else:
                logger.warning(
                    "No 'source' or 'src' column found in dataset. "
                    "Linguistic features may not work correctly."
                )

        # Same for eval dataset
        if self.eval_dataset and 'source_text' not in self.eval_dataset.column_names:
            if 'source' in self.eval_dataset.column_names:
                self.eval_dataset = self.eval_dataset.map(
                    lambda x: {'source_text': x['source']},
                    desc="Adding source_text column to eval"
                )
            elif 'src' in self.eval_dataset.column_names:
                self.eval_dataset = self.eval_dataset.map(
                    lambda x: {'source_text': x['src']},
                    desc="Adding source_text column to eval"
                )

    def _prepare_model(self):
        """Prepare model for training with linguistic features."""
        logger.info("=" * 70)
        logger.info("PREPARING MODEL FOR TRAINING WITH LINGUISTIC FEATURES")
        logger.info("=" * 70)

        if not self.model.is_loaded():
            logger.info("Loading model...")
            self.model.load()

        # Apply LoRA or prepare for full fine-tuning
        self.model.prepare_for_training(
            lora_config=self.lora_config,
            training_config=self.training_config
        )

        logger.info("=" * 70)

    def _setup_trainer(self):
        """Setup HuggingFace Seq2SeqTrainer with linguistic features support."""
        logger.info("Setting up trainer with linguistic features...")

        # Convert training config to Seq2SeqTrainingArguments
        training_args = self._create_training_arguments()

        # Create data collator (with source_text support)
        data_collator = LinguisticDataCollator(
            tokenizer=self.model.tokenizer,
            model=self.model.model,
            padding=True,
        )

        # Create compute metrics function
        def compute_metrics_fn(eval_preds):
            return compute_metrics(
                eval_preds,
                tokenizer=self.model.tokenizer,
                metric_type="both"
            )

        # Setup callbacks
        callbacks = []
        if self.training_config.early_stopping_patience > 0 and self.eval_dataset is not None:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.training_config.early_stopping_patience
                )
            )

        # Create trainer (use LinguisticSeq2SeqTrainer if linguistic features enabled)
        trainer_class = LinguisticSeq2SeqTrainer if self.linguistic_config else Seq2SeqTrainer

        self.trainer = trainer_class(
            model=self.model.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
            tokenizer=self.model.tokenizer,
            compute_metrics=compute_metrics_fn if self.eval_dataset else None,
            callbacks=callbacks,
        )

        logger.info("✓ Trainer setup complete")

    def _create_training_arguments(self) -> Seq2SeqTrainingArguments:
        """Create Seq2SeqTrainingArguments from TrainingConfig."""
        return Seq2SeqTrainingArguments(
            output_dir=str(self.training_config.output_dir),
            num_train_epochs=self.training_config.num_train_epochs,
            per_device_train_batch_size=self.training_config.per_device_train_batch_size,
            per_device_eval_batch_size=self.training_config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            learning_rate=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
            warmup_ratio=self.training_config.warmup_ratio,
            warmup_steps=self.training_config.warmup_steps,
            lr_scheduler_type=self.training_config.lr_scheduler_type,
            logging_dir=str(Path(self.training_config.output_dir) / "logs"),
            logging_steps=self.training_config.logging_steps,
            save_steps=self.training_config.save_steps,
            save_total_limit=self.training_config.save_total_limit,
            evaluation_strategy=self.training_config.evaluation_strategy,
            eval_steps=self.training_config.eval_steps,
            load_best_model_at_end=self.training_config.load_best_model_at_end,
            metric_for_best_model=self.training_config.metric_for_best_model,
            greater_is_better=self.training_config.greater_is_better,
            fp16=self.training_config.fp16,
            bf16=self.training_config.bf16,
            gradient_checkpointing=self.training_config.gradient_checkpointing,
            optim=self.training_config.optim,
            report_to=self.training_config.report_to,
            run_name=self.training_config.run_name,
            seed=self.training_config.seed,
            data_seed=self.training_config.data_seed,
            # Seq2Seq specific
            predict_with_generate=True,
            generation_max_length=self.training_config.generation_max_length,
            generation_num_beams=self.training_config.generation_num_beams,
        )

    def train(self, resume_from_checkpoint: Optional[str] = None) -> Dict[str, float]:
        """
        Train the model.

        Args:
            resume_from_checkpoint: Path to checkpoint to resume from

        Returns:
            Training metrics
        """
        logger.info("\n" + "=" * 70)
        logger.info("STARTING TRAINING")
        if self.linguistic_config:
            logger.info("Linguistic features: ENABLED")
            logger.info(f"  Parser: {self.linguistic_config.parser}")
            logger.info(f"  Features: {', '.join(self.linguistic_config.features)}")
            logger.info(f"  Integration: {self.linguistic_config.integration_method}")
        else:
            logger.info("Linguistic features: DISABLED")
        logger.info("=" * 70 + "\n")

        # Train
        train_result = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        # Save model
        self.save_model()

        # Log metrics
        metrics = train_result.metrics
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)

        logger.info("\n" + "=" * 70)
        logger.info("TRAINING COMPLETED")
        logger.info("=" * 70)

        return metrics

    def save_model(self):
        """Save the trained model."""
        logger.info(f"Saving model to {self.training_config.output_dir}")
        self.model.save(
            output_dir=str(self.training_config.output_dir),
            save_full_model=False  # Save adapter only
        )
        logger.info("✓ Model saved")


# Helper function to create appropriate trainer
def create_trainer(
    model: AbstractTrainableMTModel,
    training_config: TrainingConfig,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
    lora_config: Optional[LoRAConfig] = None,
    linguistic_config: Optional[LinguisticFeaturesConfig] = None,
) -> AbstractTrainer:
    """
    Create appropriate trainer based on configuration.

    Args:
        model: Trainable MT model
        training_config: Training configuration
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        lora_config: LoRA configuration
        linguistic_config: Linguistic features configuration

    Returns:
        Trainer instance
    """
    if linguistic_config and (
        linguistic_config.use_source_parse or linguistic_config.use_target_parse
    ):
        # Use linguistic trainer
        return LinguisticLoRATrainer(
            model=model,
            training_config=training_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            lora_config=lora_config,
            linguistic_config=linguistic_config,
        )
    else:
        # Use standard LoRA trainer
        from src.training.trainers.lora_trainer import LoRATrainer
        return LoRATrainer(
            model=model,
            training_config=training_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            lora_config=lora_config,
        )
