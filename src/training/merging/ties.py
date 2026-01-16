"""TIES-Merging: Resolving parameter conflicts when merging models.

Reference: "TIES-Merging: Resolving Interference When Merging Models"
           Yadav et al., NeurIPS 2023
           https://arxiv.org/abs/2306.01708

TIES-Merging resolves conflicts when merging fine-tuned models through:
1. TRIM: Remove redundant parameters (bottom k%)
2. ELECT SIGN: Choose majority sign for each parameter
3. DISJOINT MERGE: Average aligned parameters only

Expected improvement: +1.7 BLEU over naive averaging
"""

from typing import List, Dict, Optional
import torch
from tqdm import tqdm

from .base import AbstractMerger, MergeResult
from ..config import MergingConfig


class TIESMerger(AbstractMerger):
    """
    TIES-Merging for resolving parameter conflicts.

    Particularly effective when merging models fine-tuned on different tasks
    or with different hyperparameters that may have conflicting updates.
    """

    def __init__(self, config: MergingConfig):
        """
        Initialize TIES merger.

        Args:
            config: Merging configuration with TIES parameters
        """
        super().__init__(config)
        self.density = config.ties_density
        self.majority_sign_method = config.ties_majority_sign_method

    def merge(
        self,
        model_paths: List[str],
        base_model_path: str,
        **kwargs
    ) -> MergeResult:
        """
        Merge models using TIES-Merging.

        Args:
            model_paths: Paths to fine-tuned models
            base_model_path: Path to base model (required)
            **kwargs: Additional parameters

        Returns:
            MergeResult with merged model
        """
        print("\n" + "=" * 70)
        print("TIES-MERGING")
        print("=" * 70)
        print(f"Base model: {base_model_path}")
        print(f"Models to merge: {len(model_paths)}")
        print(f"Density (keep top): {self.density * 100}%")
        print(f"Sign method: {self.majority_sign_method}")
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

        # Apply TIES-Merging algorithm
        print("Applying TIES-Merging...")
        merged_task_vector = self._ties_merge(task_vectors)

        # Apply merged task vector to base model
        print("Applying merged task vector to base model...")
        merged_state_dict = self.apply_task_vector(
            base_state_dict,
            merged_task_vector,
            scaling=1.0
        )

        print("✓ TIES-Merging complete\n")

        return MergeResult(
            merged_state_dict=merged_state_dict,
            merge_method="ties",
            num_models_merged=len(model_paths),
            metadata={
                "base_model": base_model_path,
                "merged_models": model_paths,
                "density": self.density,
                "majority_sign_method": self.majority_sign_method,
            }
        )

    def _ties_merge(
        self,
        task_vectors: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Apply TIES-Merging algorithm.

        Steps:
        1. TRIM: Remove bottom (1-density)% of parameters
        2. ELECT SIGN: Choose majority sign
        3. DISJOINT MERGE: Average aligned parameters

        Args:
            task_vectors: List of task vectors (deltas)

        Returns:
            Merged task vector
        """
        merged_vector = {}

        # Get all parameter keys
        param_keys = list(task_vectors[0].keys())

        for key in tqdm(param_keys, desc="Merging parameters"):
            # Stack parameters from all models
            params = [tv[key] for tv in task_vectors]
            params_stacked = torch.stack(params, dim=0)

            # Step 1: TRIM - Keep top density% of parameters by magnitude
            trimmed_params = self._trim_parameters(params_stacked)

            # Step 2: ELECT SIGN - Determine majority sign
            elected_sign = self._elect_sign(trimmed_params)

            # Step 3: DISJOINT MERGE - Average aligned parameters
            merged_param = self._disjoint_merge(trimmed_params, elected_sign)

            merged_vector[key] = merged_param

        return merged_vector

    def _trim_parameters(
        self,
        params: torch.Tensor
    ) -> torch.Tensor:
        """
        TRIM: Keep top density% of parameters by absolute value.

        Args:
            params: Stacked parameters [num_models, *param_shape]

        Returns:
            Trimmed parameters (bottom values set to 0)
        """
        # Compute absolute values
        abs_params = torch.abs(params)

        # Compute threshold per model
        num_models = params.shape[0]
        trimmed = torch.zeros_like(params)

        for i in range(num_models):
            model_params = params[i]
            abs_model_params = abs_params[i]

            # Flatten to compute percentile
            flat_abs = abs_model_params.flatten()

            # Compute threshold (keep top density%)
            k = int(len(flat_abs) * (1 - self.density))
            if k > 0:
                threshold = torch.kthvalue(flat_abs, k).values
            else:
                threshold = 0.0

            # Keep only parameters above threshold
            mask = abs_model_params > threshold
            trimmed[i] = model_params * mask

        return trimmed

    def _elect_sign(self, params: torch.Tensor) -> torch.Tensor:
        """
        ELECT SIGN: Determine majority sign for each parameter.

        Args:
            params: Trimmed parameters [num_models, *param_shape]

        Returns:
            Elected signs (+1 or -1)
        """
        if self.majority_sign_method == "total":
            # Method 1: Sum of values (considering magnitude)
            total = torch.sum(params, dim=0)
            elected_sign = torch.sign(total)

        elif self.majority_sign_method == "frequency":
            # Method 2: Frequency of signs (binary vote)
            signs = torch.sign(params)
            # Count positive vs negative
            positive_count = (signs > 0).sum(dim=0).float()
            negative_count = (signs < 0).sum(dim=0).float()

            elected_sign = torch.where(
                positive_count >= negative_count,
                torch.ones_like(positive_count),
                -torch.ones_like(negative_count)
            )

        else:
            raise ValueError(f"Unknown sign method: {self.majority_sign_method}")

        # Handle zeros (no majority)
        elected_sign = torch.where(
            elected_sign == 0,
            torch.ones_like(elected_sign),  # Default to positive
            elected_sign
        )

        return elected_sign

    def _disjoint_merge(
        self,
        params: torch.Tensor,
        elected_sign: torch.Tensor
    ) -> torch.Tensor:
        """
        DISJOINT MERGE: Average parameters with matching signs.

        Args:
            params: Trimmed parameters [num_models, *param_shape]
            elected_sign: Elected signs for each parameter

        Returns:
            Merged parameters
        """
        # Get signs of each parameter
        param_signs = torch.sign(params)

        # Mask for parameters matching elected sign
        # Shape: [num_models, *param_shape]
        aligned_mask = (param_signs == elected_sign.unsqueeze(0)).float()

        # Set non-aligned parameters to 0
        aligned_params = params * aligned_mask

        # Average aligned parameters
        # Sum aligned values and divide by number of aligned models per parameter
        sum_aligned = torch.sum(aligned_params, dim=0)
        count_aligned = torch.sum(aligned_mask, dim=0)

        # Avoid division by zero
        count_aligned = torch.where(
            count_aligned == 0,
            torch.ones_like(count_aligned),
            count_aligned
        )

        merged = sum_aligned / count_aligned

        return merged

    @staticmethod
    def analyze_conflicts(
        task_vectors: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, Any]:
        """
        Analyze parameter conflicts between models.

        Useful for deciding whether TIES-Merging is beneficial.

        Args:
            task_vectors: List of task vectors

        Returns:
            Conflict analysis results
        """
        print("Analyzing parameter conflicts...")

        total_params = 0
        conflicting_params = 0
        conflict_magnitude = []

        param_keys = list(task_vectors[0].keys())

        for key in param_keys:
            params = torch.stack([tv[key] for tv in task_vectors], dim=0)

            # Get signs
            signs = torch.sign(params)

            # Check for conflicts (different signs)
            # For each position, count unique non-zero signs
            non_zero_mask = (params != 0)
            masked_signs = signs * non_zero_mask

            # Reshape for conflict detection
            flat_signs = masked_signs.reshape(len(task_vectors), -1)

            for pos in range(flat_signs.shape[1]):
                pos_signs = flat_signs[:, pos]
                non_zero_signs = pos_signs[pos_signs != 0]

                if len(non_zero_signs) > 1:
                    unique_signs = torch.unique(non_zero_signs)
                    if len(unique_signs) > 1:
                        conflicting_params += 1
                        # Measure conflict magnitude
                        conflict_magnitude.append(
                            torch.std(params[:, pos].flatten()).item()
                        )

                total_params += 1

        conflict_rate = conflicting_params / total_params if total_params > 0 else 0
        avg_conflict_magnitude = sum(conflict_magnitude) / len(conflict_magnitude) if conflict_magnitude else 0

        results = {
            "total_parameters": total_params,
            "conflicting_parameters": conflicting_params,
            "conflict_rate": conflict_rate,
            "avg_conflict_magnitude": avg_conflict_magnitude,
            "recommendation": "Use TIES-Merging" if conflict_rate > 0.1 else "Simple averaging may suffice"
        }

        print(f"\nConflict Analysis:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Conflicting: {conflicting_params:,} ({conflict_rate*100:.1f}%)")
        print(f"  Avg conflict magnitude: {avg_conflict_magnitude:.4f}")
        print(f"  Recommendation: {results['recommendation']}\n")

        return results
