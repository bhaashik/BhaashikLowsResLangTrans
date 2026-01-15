"""Abstract base trainer."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from pathlib import Path


class AbstractTrainer(ABC):
    """
    Abstract base class for all trainers.

    Provides a unified interface for training MT models.
    """

    def __init__(
        self,
        model: Any,
        training_config: Any,
        data_config: Optional[Any] = None
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train
            training_config: Training configuration
            data_config: Data configuration (optional)
        """
        self.model = model
        self.training_config = training_config
        self.data_config = data_config
        self.trainer = None  # HuggingFace Trainer instance
        self.training_results = None

    @abstractmethod
    def train(self) -> Dict[str, Any]:
        """
        Run training loop.

        Returns:
            Training results dictionary
        """
        pass

    @abstractmethod
    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate model.

        Returns:
            Evaluation metrics
        """
        pass

    def save_model(self, output_dir: Optional[str] = None):
        """
        Save trained model.

        Args:
            output_dir: Output directory (uses training_config.output_dir if None)
        """
        save_dir = output_dir or self.training_config.output_dir
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        print(f"Saving model to {save_dir}")
        self.model.save(save_dir)
        print("✓ Model saved successfully")

    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get training summary.

        Returns:
            Summary dictionary with key metrics
        """
        if self.training_results is None:
            return {"status": "not_trained"}

        return {
            "status": "completed",
            "train_loss": self.training_results.get("train_loss"),
            "eval_loss": self.training_results.get("eval_loss"),
            "best_metric": self.training_results.get("best_metric"),
            "total_steps": self.training_results.get("global_step"),
        }

    def print_training_summary(self):
        """Print formatted training summary."""
        summary = self.get_training_summary()

        print("\n" + "=" * 70)
        print("TRAINING SUMMARY")
        print("=" * 70)

        for key, value in summary.items():
            print(f"{key}: {value}")

        print("=" * 70 + "\n")
