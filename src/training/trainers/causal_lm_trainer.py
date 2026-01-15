"""Causal LM trainer for instruction-following LLMs."""

from typing import Optional, Dict, Any, List
from pathlib import Path
from datasets import Dataset
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)
import torch

from .base import AbstractTrainer
from ..config import TrainingConfig, LoRAConfig
from ..models.hf_llm import HuggingFaceLLMModel


class CausalLMTrainer(AbstractTrainer):
    """
    Trainer for fine-tuning causal LLMs for translation.

    Supports Llama, Mistral, Mixtral, and other instruction-following models.
    Uses instruction-following format with label masking for prompts.
    """

    def __init__(
        self,
        model: HuggingFaceLLMModel,
        training_config: TrainingConfig,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        lora_config: Optional[LoRAConfig] = None
    ):
        """
        Initialize Causal LM trainer.

        Args:
            model: HuggingFace LLM model
            training_config: Training configuration
            train_dataset: Training dataset (should have "source" and "target" columns)
            eval_dataset: Evaluation dataset (optional)
            lora_config: LoRA configuration (recommended for LLMs)
        """
        super().__init__(model, training_config)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.lora_config = lora_config

        # Prepare model for training
        self._prepare_model()

        # Preprocess datasets
        self._preprocess_datasets()

        # Setup trainer
        self._setup_trainer()

    def _prepare_model(self):
        """Prepare model for training (apply LoRA recommended for LLMs)."""
        print("\n" + "=" * 70)
        print("PREPARING LLM FOR TRAINING")
        print("=" * 70)

        if not self.model.is_loaded():
            print("Loading model...")
            self.model.load()

        # Apply LoRA (highly recommended for LLMs)
        if self.lora_config is None:
            print("WARNING: No LoRA config provided. Using default LoRA settings.")
            print("Full fine-tuning of LLMs requires significant memory!")
            from ..config import LoRAConfig
            self.lora_config = LoRAConfig.create_preset("llm")

        self.model.prepare_for_training(
            lora_config=self.lora_config,
            training_config=self.training_config
        )

        print("=" * 70 + "\n")

    def _preprocess_datasets(self):
        """Preprocess datasets for causal LM training with label masking."""
        print("Preprocessing datasets for instruction-following format...")

        # Process train dataset
        self.train_dataset = self.train_dataset.map(
            self._format_example,
            remove_columns=self.train_dataset.column_names,
            desc="Formatting train dataset"
        )

        # Process eval dataset if available
        if self.eval_dataset is not None:
            self.eval_dataset = self.eval_dataset.map(
                self._format_example,
                remove_columns=self.eval_dataset.column_names,
                desc="Formatting eval dataset"
            )

        print("✓ Datasets preprocessed")

    def _format_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format training example with label masking.

        Args:
            example: Raw example with "source" and "target" fields

        Returns:
            Formatted example with input_ids, attention_mask, labels
        """
        source_text = example["source"]
        target_text = example["target"]

        # Use model's format_training_example method
        formatted = self.model.format_training_example(
            source_text=source_text,
            target_text=target_text
        )

        return formatted

    def _setup_trainer(self):
        """Setup HuggingFace Trainer for causal LM."""
        print("Setting up trainer...")

        # Convert training config to TrainingArguments
        training_args = self._create_training_arguments()

        # Create data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.model.tokenizer,
            mlm=False,  # Causal LM, not masked LM
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
        self.trainer = Trainer(
            model=self.model.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
            callbacks=callbacks,
        )

        print("✓ Trainer setup complete")

    def _create_training_arguments(self) -> TrainingArguments:
        """Create TrainingArguments from config."""
        config = self.training_config

        args = TrainingArguments(
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

            # Memory optimization (important for LLMs)
            gradient_checkpointing=True,  # Always enable for LLMs

            # Dataloader
            dataloader_num_workers=config.dataloader_num_workers,
            dataloader_pin_memory=config.dataloader_pin_memory,

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
        print("STARTING LLM TRAINING")
        print("=" * 70)
        print(f"Output directory: {self.training_config.output_dir}")
        print(f"Training samples: {len(self.train_dataset):,}")
        if self.eval_dataset:
            print(f"Evaluation samples: {len(self.eval_dataset):,}")
        print(f"Epochs: {self.training_config.num_train_epochs}")
        print(f"Batch size: {self.training_config.per_device_train_batch_size}")
        print(f"Learning rate: {self.training_config.learning_rate}")
        print(f"LoRA enabled: {self.lora_config is not None}")
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
        self.trainer.save_metrics("train", train_result.metrics)

        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"Final train loss: {train_result.metrics.get('train_loss', 'N/A'):.4f}")
        if self.eval_dataset:
            eval_results = self.evaluate()
            print(f"Final eval loss: {eval_results.get('eval_loss', 'N/A'):.4f}")
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
        self.trainer.save_metrics("eval", eval_results)

        return eval_results

    def generate_translations(
        self,
        test_texts: List[str],
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
        **generation_kwargs
    ) -> List[str]:
        """
        Generate translations for test texts.

        Args:
            test_texts: List of source texts
            source_lang: Source language (overrides model config)
            target_lang: Target language (overrides model config)
            **generation_kwargs: Additional generation parameters

        Returns:
            List of translations
        """
        print(f"\nGenerating translations for {len(test_texts)} texts...")

        translations = self.model.translate(
            texts=test_texts,
            source_lang=source_lang,
            target_lang=target_lang,
            **generation_kwargs
        )

        return translations
