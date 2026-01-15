"""LoRA trainer for seq2seq models."""

from typing import Optional, Dict, Any
from pathlib import Path
from datasets import Dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)

from .base import AbstractTrainer
from .metrics import compute_metrics
from ..config import TrainingConfig, LoRAConfig
from ..models.base import AbstractTrainableMTModel


class LoRATrainer(AbstractTrainer):
    """
    Trainer for fine-tuning seq2seq models with LoRA.

    Supports NLLB, mBART, mT5, IndicTrans2, and other seq2seq architectures.
    Optimized for V100 GPUs and memory-efficient training.
    """

    def __init__(
        self,
        model: AbstractTrainableMTModel,
        training_config: TrainingConfig,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        lora_config: Optional[LoRAConfig] = None
    ):
        """
        Initialize LoRA trainer.

        Args:
            model: Trainable MT model
            training_config: Training configuration
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            lora_config: LoRA configuration (None for full fine-tuning)
        """
        super().__init__(model, training_config)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.lora_config = lora_config

        # Prepare model for training
        self._prepare_model()

        # Setup trainer
        self._setup_trainer()

    def _prepare_model(self):
        """Prepare model for training (apply LoRA or full fine-tuning)."""
        print("\n" + "=" * 70)
        print("PREPARING MODEL FOR TRAINING")
        print("=" * 70)

        if not self.model.is_loaded():
            print("Loading model...")
            self.model.load()

        # Apply LoRA or prepare for full fine-tuning
        self.model.prepare_for_training(
            lora_config=self.lora_config,
            training_config=self.training_config
        )

        print("=" * 70 + "\n")

    def _setup_trainer(self):
        """Setup HuggingFace Seq2SeqTrainer."""
        print("Setting up trainer...")

        # Convert training config to Seq2SeqTrainingArguments
        training_args = self._create_training_arguments()

        # Create data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.model.tokenizer,
            model=self.model.model,
            padding=True,
        )

        # Create compute metrics function
        def compute_metrics_fn(eval_preds):
            return compute_metrics(
                eval_preds,
                tokenizer=self.model.tokenizer,
                metric_type="both"  # Compute both BLEU and chrF
            )

        # Setup callbacks
        callbacks = []
        if self.training_config.early_stopping_patience > 0 and self.eval_dataset is not None:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.training_config.early_stopping_patience,
                    early_stopping_threshold=self.training_config.early_stopping_threshold,
                )
            )

        # Create trainer
        self.trainer = Seq2SeqTrainer(
            model=self.model.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics_fn if self.eval_dataset else None,
            callbacks=callbacks,
        )

        print("✓ Trainer setup complete")

    def _create_training_arguments(self) -> Seq2SeqTrainingArguments:
        """Create Seq2SeqTrainingArguments from config."""
        config = self.training_config

        args = Seq2SeqTrainingArguments(
            # Output
            output_dir=config.output_dir,
            run_name=config.run_name,
            logging_dir=config.logging_dir,

            # Training
            num_train_epochs=config.num_train_epochs,
            per_device_train_batch_size=config.per_device_train_batch_size,
            per_device_eval_batch_size=config.per_device_eval_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            max_grad_norm=config.max_grad_norm,

            # Optimizer
            optim=config.optim,
            adam_beta1=config.adam_beta1,
            adam_beta2=config.adam_beta2,
            adam_epsilon=config.adam_epsilon,

            # LR scheduler
            lr_scheduler_type=config.lr_scheduler_type,
            warmup_steps=config.warmup_steps,
            warmup_ratio=config.warmup_ratio,

            # Evaluation
            evaluation_strategy=config.evaluation_strategy,
            eval_steps=config.eval_steps if config.evaluation_strategy == "steps" else None,
            eval_accumulation_steps=config.eval_accumulation_steps,

            # Saving
            save_strategy=config.save_strategy,
            save_steps=config.save_steps if config.save_strategy == "steps" else None,
            save_total_limit=config.save_total_limit,
            load_best_model_at_end=config.load_best_model_at_end,
            metric_for_best_model=config.metric_for_best_model,
            greater_is_better=config.greater_is_better,

            # Logging
            logging_steps=config.logging_steps,
            logging_first_step=config.logging_first_step,
            logging_strategy=config.logging_strategy,
            report_to=config.report_to,

            # Mixed precision
            fp16=config.fp16,
            bf16=config.bf16,
            fp16_opt_level=config.fp16_opt_level,
            fp16_backend=config.fp16_backend,

            # Memory optimization
            gradient_checkpointing=config.gradient_checkpointing,

            # Dataloader
            dataloader_num_workers=config.dataloader_num_workers,
            dataloader_pin_memory=config.dataloader_pin_memory,

            # Generation
            predict_with_generate=config.predict_with_generate,
            generation_max_length=config.max_new_tokens,
            generation_num_beams=config.num_beams,

            # Reproducibility
            seed=config.seed,
            data_seed=config.data_seed,

            # Device
            local_rank=config.local_rank,
            ddp_find_unused_parameters=config.ddp_find_unused_parameters,

            # Resume
            resume_from_checkpoint=config.resume_from_checkpoint,
            ignore_data_skip=config.ignore_data_skip,

            # Advanced
            label_smoothing_factor=config.label_smoothing_factor,
            include_inputs_for_metrics=config.include_inputs_for_metrics,
            auto_find_batch_size=config.auto_find_batch_size,
        )

        return args

    def train(self) -> Dict[str, Any]:
        """
        Run training loop.

        Returns:
            Training results dictionary
        """
        print("\n" + "=" * 70)
        print("STARTING TRAINING")
        print("=" * 70)
        print(f"Output directory: {self.training_config.output_dir}")
        print(f"Training samples: {len(self.train_dataset):,}")
        if self.eval_dataset:
            print(f"Evaluation samples: {len(self.eval_dataset):,}")
        print(f"Epochs: {self.training_config.num_train_epochs}")
        print(f"Batch size: {self.training_config.per_device_train_batch_size}")
        print(f"Learning rate: {self.training_config.learning_rate}")
        print("=" * 70 + "\n")

        # Train
        train_result = self.trainer.train(
            resume_from_checkpoint=self.training_config.resume_from_checkpoint
        )

        # Save results
        self.training_results = train_result.metrics

        # Save model
        self.save_model()

        # Save training metrics
        metrics_path = Path(self.training_config.output_dir) / "train_results.json"
        self.trainer.save_metrics("train", train_result.metrics)

        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"Final train loss: {train_result.metrics.get('train_loss', 'N/A'):.4f}")
        if self.eval_dataset:
            eval_results = self.evaluate()
            print(f"Final eval loss: {eval_results.get('eval_loss', 'N/A'):.4f}")
            print(f"Final BLEU: {eval_results.get('eval_bleu', 'N/A'):.2f}")
            print(f"Final chrF: {eval_results.get('eval_chrf', 'N/A'):.2f}")
        print("=" * 70 + "\n")

        return train_result.metrics

    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate model.

        Returns:
            Evaluation metrics
        """
        if self.eval_dataset is None:
            print("No evaluation dataset provided")
            return {}

        print("\nRunning evaluation...")
        eval_results = self.trainer.evaluate()

        # Save evaluation metrics
        metrics_path = Path(self.training_config.output_dir) / "eval_results.json"
        self.trainer.save_metrics("eval", eval_results)

        return eval_results

    def predict(self, test_dataset: Dataset) -> Dict[str, Any]:
        """
        Generate predictions on test dataset.

        Args:
            test_dataset: Test dataset

        Returns:
            Prediction results with metrics
        """
        print("\nGenerating predictions...")

        predictions = self.trainer.predict(test_dataset)

        # Decode predictions
        decoded_preds = self.model.tokenizer.batch_decode(
            predictions.predictions,
            skip_special_tokens=True
        )

        # Decode labels if available
        if predictions.label_ids is not None:
            labels = predictions.label_ids
            labels = labels.where(labels != -100, self.model.tokenizer.pad_token_id)
            decoded_labels = self.model.tokenizer.batch_decode(
                labels,
                skip_special_tokens=True
            )
        else:
            decoded_labels = None

        return {
            "predictions": decoded_preds,
            "labels": decoded_labels,
            "metrics": predictions.metrics if hasattr(predictions, "metrics") else None,
        }
