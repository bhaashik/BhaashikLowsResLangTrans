"""Base classes for model merging."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass
import torch


@dataclass
class MergeResult:
    """Result of model merging operation."""

    merged_state_dict: Dict[str, torch.Tensor]
    merge_method: str
    num_models_merged: int
    metadata: Dict[str, Any]

    def save_merged_model(
        self,
        output_dir: str,
        model_class,
        tokenizer=None
    ):
        """
        Save merged model to disk.

        Args:
            output_dir: Output directory
            model_class: Model class to instantiate
            tokenizer: Tokenizer to save (optional)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"Saving merged model to {output_dir}")

        # Load merged state dict into model
        model = model_class.from_pretrained(
            self.metadata.get("base_model"),
            state_dict=self.merged_state_dict,
        )

        # Save model
        model.save_pretrained(output_dir)

        # Save tokenizer if provided
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)

        # Save merge metadata
        import json
        metadata_path = output_path / "merge_metadata.json"
        with open(metadata_path, 'w') as f:
            # Convert tensors to serializable format
            serializable_metadata = {
                k: v for k, v in self.metadata.items()
                if not isinstance(v, torch.Tensor)
            }
            json.dump(serializable_metadata, f, indent=2)

        print("✓ Merged model saved successfully")


class AbstractMerger(ABC):
    """
    Abstract base class for model merging strategies.

    All merging methods should inherit from this class.
    """

    def __init__(self, config: Any):
        """
        Initialize merger.

        Args:
            config: Merging configuration
        """
        self.config = config

    @abstractmethod
    def merge(
        self,
        model_paths: List[str],
        base_model_path: Optional[str] = None,
        **kwargs
    ) -> MergeResult:
        """
        Merge multiple models.

        Args:
            model_paths: Paths to models to merge
            base_model_path: Path to base model (for delta-based methods)
            **kwargs: Additional merging parameters

        Returns:
            MergeResult with merged state dict
        """
        pass

    def load_model_state_dict(self, model_path: str) -> Dict[str, torch.Tensor]:
        """
        Load model state dict from path.

        Args:
            model_path: Path to model

        Returns:
            State dict
        """
        from transformers import AutoModel

        print(f"Loading model from {model_path}")

        # Load model
        model = AutoModel.from_pretrained(model_path)
        state_dict = model.state_dict()

        # Free memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return state_dict

    def compute_task_vector(
        self,
        finetuned_state_dict: Dict[str, torch.Tensor],
        base_state_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute task vector (delta) between fine-tuned and base model.

        Args:
            finetuned_state_dict: Fine-tuned model state dict
            base_state_dict: Base model state dict

        Returns:
            Task vector (delta)
        """
        task_vector = {}

        for key in finetuned_state_dict.keys():
            if key in base_state_dict:
                task_vector[key] = finetuned_state_dict[key] - base_state_dict[key]

        return task_vector

    def apply_task_vector(
        self,
        base_state_dict: Dict[str, torch.Tensor],
        task_vector: Dict[str, torch.Tensor],
        scaling: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Apply task vector to base model.

        Args:
            base_state_dict: Base model state dict
            task_vector: Task vector to apply
            scaling: Scaling factor for task vector

        Returns:
            New state dict
        """
        new_state_dict = {}

        for key in base_state_dict.keys():
            if key in task_vector:
                new_state_dict[key] = base_state_dict[key] + scaling * task_vector[key]
            else:
                new_state_dict[key] = base_state_dict[key].clone()

        return new_state_dict


def load_multiple_models(
    model_paths: List[str],
    device: str = "cpu"
) -> List[Dict[str, torch.Tensor]]:
    """
    Load multiple models efficiently.

    Args:
        model_paths: List of model paths
        device: Device to load models on

    Returns:
        List of state dicts
    """
    state_dicts = []

    for path in model_paths:
        merger = AbstractMerger(None)  # Dummy instance for loading
        state_dict = merger.load_model_state_dict(path)

        # Move to device
        if device != "cpu":
            state_dict = {k: v.to(device) for k, v in state_dict.items()}

        state_dicts.append(state_dict)

    return state_dicts
