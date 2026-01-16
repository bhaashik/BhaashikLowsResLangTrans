"""DARE: Drop And REscale for efficient model merging.

Reference: "Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch"
           Yu et al., 2024
           https://arxiv.org/abs/2311.03099

DARE randomly drops a large percentage (90-99%) of fine-tuning updates (deltas)
and rescales the remaining parameters. This surprisingly improves merged model quality.

Key insight: Most fine-tuning updates are redundant. Dropping them reduces noise.

Expected improvement: +1.9 BLEU over naive averaging
"""

from typing import List, Dict, Optional
import torch
import numpy as np
from tqdm import tqdm

from .base import AbstractMerger, MergeResult
from ..config import MergingConfig


class DAREMerger(AbstractMerger):
    """
    DARE (Drop And REscale) merger.

    Randomly drops most delta parameters and rescales the rest.
    Counterintuitively, this often improves quality by reducing noise.
    """

    def __init__(self, config: MergingConfig):
        """
        Initialize DARE merger.

        Args:
            config: Merging configuration with DARE parameters
        """
        super().__init__(config)
        self.drop_rate = config.dare_drop_rate
        self.rescale_mode = config.dare_rescale_mode
        self.rescale_factor = config.dare_rescale_factor

    def merge(
        self,
        model_paths: List[str],
        base_model_path: str,
        **kwargs
    ) -> MergeResult:
        """
        Merge models using DARE.

        Args:
            model_paths: Paths to fine-tuned models
            base_model_path: Path to base model (required)
            **kwargs: Additional parameters

        Returns:
            MergeResult with merged model
        """
        print("\n" + "=" * 70)
        print("DARE (Drop And REscale)")
        print("=" * 70)
        print(f"Base model: {base_model_path}")
        print(f"Models to merge: {len(model_paths)}")
        print(f"Drop rate: {self.drop_rate * 100}%")
        print(f"Rescale mode: {self.rescale_mode}")
        print("=" * 70 + "\n")

        # Load base model
        print("Loading base model...")
        base_state_dict = self.load_model_state_dict(base_model_path)

        # Load fine-tuned models
        print("Loading fine-tuned models...")
        finetuned_state_dicts = [
            self.load_model_state_dict(path) for path in model_paths
        ]

        # Compute task vectors (deltas)
        print("Computing task vectors...")
        task_vectors = [
            self.compute_task_vector(ft_dict, base_state_dict)
            for ft_dict in finetuned_state_dicts
        ]

        # Apply DARE to each task vector
        print(f"Applying DARE (dropping {self.drop_rate*100}% of parameters)...")
        dare_vectors = [
            self._apply_dare(tv) for tv in tqdm(task_vectors, desc="DARE per model")
        ]

        # Average DARE vectors
        print("Averaging DARE vectors...")
        merged_task_vector = self._average_vectors(dare_vectors)

        # Apply merged task vector to base model
        print("Applying merged task vector to base model...")
        merged_state_dict = self.apply_task_vector(
            base_state_dict,
            merged_task_vector,
            scaling=1.0
        )

        print("✓ DARE merging complete\n")

        return MergeResult(
            merged_state_dict=merged_state_dict,
            merge_method="dare",
            num_models_merged=len(model_paths),
            metadata={
                "base_model": base_model_path,
                "merged_models": model_paths,
                "drop_rate": self.drop_rate,
                "rescale_mode": self.rescale_mode,
            }
        )

    def _apply_dare(
        self,
        task_vector: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Apply DARE to a single task vector.

        Args:
            task_vector: Task vector (delta from base model)

        Returns:
            DARE-processed task vector
        """
        dare_vector = {}

        for key, delta in task_vector.items():
            # Drop parameters randomly
            dropped_delta = self._drop_parameters(delta)

            # Rescale remaining parameters
            rescaled_delta = self._rescale_parameters(dropped_delta, delta)

            dare_vector[key] = rescaled_delta

        return dare_vector

    def _drop_parameters(self, delta: torch.Tensor) -> torch.Tensor:
        """
        Randomly drop parameters.

        Args:
            delta: Parameter delta

        Returns:
            Delta with dropped parameters set to 0
        """
        # Create random mask
        keep_prob = 1 - self.drop_rate
        mask = torch.bernoulli(torch.full_like(delta, keep_prob))

        # Apply mask
        dropped = delta * mask

        return dropped

    def _rescale_parameters(
        self,
        dropped_delta: torch.Tensor,
        original_delta: torch.Tensor
    ) -> torch.Tensor:
        """
        Rescale parameters after dropping.

        Args:
            dropped_delta: Delta after dropping
            original_delta: Original delta (for adaptive scaling)

        Returns:
            Rescaled delta
        """
        if self.rescale_mode == "constant":
            # Constant rescaling by 1/(1-drop_rate)
            scale = self.rescale_factor / (1 - self.drop_rate)
            rescaled = dropped_delta * scale

        elif self.rescale_mode == "adaptive":
            # Adaptive rescaling to preserve magnitude
            # Scale factor computed per parameter to match original magnitude

            original_norm = torch.norm(original_delta)
            dropped_norm = torch.norm(dropped_delta)

            if dropped_norm > 0:
                scale = original_norm / dropped_norm
                rescaled = dropped_delta * scale
            else:
                rescaled = dropped_delta

        elif self.rescale_mode == "layer_wise":
            # Layer-wise adaptive scaling
            # Reshape to 2D for layer-wise operations
            original_shape = original_delta.shape

            if len(original_shape) > 1:
                # Flatten all dims except first (treat as layers)
                orig_2d = original_delta.reshape(original_shape[0], -1)
                drop_2d = dropped_delta.reshape(original_shape[0], -1)

                # Compute scale per layer
                orig_norms = torch.norm(orig_2d, dim=1, keepdim=True)
                drop_norms = torch.norm(drop_2d, dim=1, keepdim=True)

                # Avoid division by zero
                scales = torch.where(
                    drop_norms > 0,
                    orig_norms / drop_norms,
                    torch.ones_like(orig_norms)
                )

                # Apply scales
                rescaled_2d = drop_2d * scales

                # Reshape back
                rescaled = rescaled_2d.reshape(original_shape)
            else:
                # Fallback to adaptive for 1D tensors
                original_norm = torch.norm(original_delta)
                dropped_norm = torch.norm(dropped_delta)

                if dropped_norm > 0:
                    scale = original_norm / dropped_norm
                    rescaled = dropped_delta * scale
                else:
                    rescaled = dropped_delta

        else:
            raise ValueError(f"Unknown rescale mode: {self.rescale_mode}")

        return rescaled

    def _average_vectors(
        self,
        vectors: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Average multiple task vectors.

        Args:
            vectors: List of task vectors

        Returns:
            Averaged task vector
        """
        averaged = {}

        param_keys = list(vectors[0].keys())

        for key in param_keys:
            params = torch.stack([v[key] for v in vectors], dim=0)
            averaged[key] = torch.mean(params, dim=0)

        return averaged

    @staticmethod
    def find_optimal_drop_rate(
        model_paths: List[str],
        base_model_path: str,
        eval_dataset,
        eval_metric_fn,
        drop_rates: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Find optimal drop rate by grid search.

        Args:
            model_paths: Paths to models
            base_model_path: Base model path
            eval_dataset: Evaluation dataset
            eval_metric_fn: Function to compute evaluation metric
            drop_rates: Drop rates to try (default: [0.7, 0.8, 0.9, 0.95, 0.99])

        Returns:
            Dictionary with best drop rate and scores
        """
        from ..config import MergingConfig

        if drop_rates is None:
            drop_rates = [0.7, 0.8, 0.9, 0.95, 0.99]

        print("\n" + "=" * 70)
        print("FINDING OPTIMAL DARE DROP RATE")
        print("=" * 70)

        results = {}

        for drop_rate in drop_rates:
            print(f"\nTrying drop_rate = {drop_rate}")

            # Create config
            config = MergingConfig(
                method="dare",
                model_paths=model_paths,
                base_model_path=base_model_path,
                dare_drop_rate=drop_rate
            )

            # Merge
            merger = DAREMerger(config)
            merge_result = merger.merge(model_paths, base_model_path)

            # Evaluate
            score = eval_metric_fn(merge_result.merged_state_dict, eval_dataset)

            results[drop_rate] = score
            print(f"  Score: {score:.4f}")

        # Find best
        best_drop_rate = max(results, key=results.get)
        best_score = results[best_drop_rate]

        print("\n" + "=" * 70)
        print(f"Best drop rate: {best_drop_rate} (score: {best_score:.4f})")
        print("=" * 70 + "\n")

        return {
            "best_drop_rate": best_drop_rate,
            "best_score": best_score,
            "all_scores": results
        }
