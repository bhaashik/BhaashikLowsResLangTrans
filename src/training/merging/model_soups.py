"""Model Soups: Averaging checkpoints for improved performance.

Reference: "Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time"
           Wortsman et al., ICML 2022
           https://arxiv.org/abs/2203.05482

Model Soups averages weights from:
1. Multiple checkpoints from same training run
2. Models trained with different hyperparameters
3. Models trained with different random seeds

Two variants:
- Uniform Soup: Simple average of all models
- Greedy Soup: Iteratively add models that improve validation performance

Expected improvement: +0.6-0.8 BLEU over single best model
"""

from typing import List, Dict, Optional, Callable
import torch
from tqdm import tqdm

from .base import AbstractMerger, MergeResult
from ..config import MergingConfig


class ModelSoups(AbstractMerger):
    """
    Model Soups for checkpoint and multi-run averaging.

    Simple yet effective method that averages model weights.
    """

    def __init__(self, config: MergingConfig):
        """
        Initialize Model Soups merger.

        Args:
            config: Merging configuration
        """
        super().__init__(config)
        self.soup_method = config.soup_method
        self.max_models = config.soup_max_models

    def merge(
        self,
        model_paths: List[str],
        base_model_path: Optional[str] = None,
        eval_dataset=None,
        eval_metric_fn: Optional[Callable] = None,
        **kwargs
    ) -> MergeResult:
        """
        Create model soup.

        Args:
            model_paths: Paths to models/checkpoints to merge
            base_model_path: Not used for soups (models should be from same base)
            eval_dataset: Evaluation dataset (required for greedy soup)
            eval_metric_fn: Evaluation metric function (required for greedy soup)
            **kwargs: Additional parameters

        Returns:
            MergeResult with souped model
        """
        print("\n" + "=" * 70)
        print(f"MODEL SOUPS ({self.soup_method.upper()})")
        print("=" * 70)
        print(f"Models to soup: {len(model_paths)}")
        if self.max_models:
            print(f"Max models: {self.max_models}")
        print("=" * 70 + "\n")

        if self.soup_method == "uniform":
            merged_state_dict = self._uniform_soup(model_paths)
        elif self.soup_method == "greedy":
            if eval_dataset is None or eval_metric_fn is None:
                raise ValueError("Greedy soup requires eval_dataset and eval_metric_fn")
            merged_state_dict, selected_models = self._greedy_soup(
                model_paths, eval_dataset, eval_metric_fn
            )
        else:
            raise ValueError(f"Unknown soup method: {self.soup_method}")

        print("✓ Model soup complete\n")

        metadata = {
            "souped_models": model_paths,
            "soup_method": self.soup_method,
        }

        if self.soup_method == "greedy":
            metadata["selected_models"] = selected_models

        return MergeResult(
            merged_state_dict=merged_state_dict,
            merge_method=f"model_soups_{self.soup_method}",
            num_models_merged=len(model_paths),
            metadata=metadata
        )

    def _uniform_soup(
        self,
        model_paths: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        Uniform soup: Simple average of all models.

        Args:
            model_paths: Paths to models

        Returns:
            Averaged state dict
        """
        print("Creating uniform soup...")

        # Load all models
        state_dicts = []
        for path in tqdm(model_paths, desc="Loading models"):
            state_dict = self.load_model_state_dict(path)
            state_dicts.append(state_dict)

        # Average parameters
        print("Averaging parameters...")
        averaged = {}

        param_keys = list(state_dicts[0].keys())

        for key in tqdm(param_keys, desc="Averaging"):
            params = torch.stack([sd[key] for sd in state_dicts], dim=0)
            averaged[key] = torch.mean(params, dim=0)

        return averaged

    def _greedy_soup(
        self,
        model_paths: List[str],
        eval_dataset,
        eval_metric_fn: Callable
    ) -> tuple:
        """
        Greedy soup: Iteratively add models that improve validation performance.

        Args:
            model_paths: Paths to models
            eval_dataset: Evaluation dataset
            eval_metric_fn: Function to evaluate model (higher is better)

        Returns:
            Tuple of (souped_state_dict, selected_model_paths)
        """
        print("Creating greedy soup...")

        # Load all models
        print("Loading candidate models...")
        candidates = {}
        candidate_scores = {}

        for path in tqdm(model_paths, desc="Loading & evaluating"):
            state_dict = self.load_model_state_dict(path)
            candidates[path] = state_dict

            # Evaluate individual model
            score = eval_metric_fn(state_dict, eval_dataset)
            candidate_scores[path] = score
            print(f"  {path}: {score:.4f}")

        # Start with best individual model
        best_path = max(candidate_scores, key=candidate_scores.get)
        print(f"\nStarting with best model: {best_path} (score: {candidate_scores[best_path]:.4f})")

        selected_models = [best_path]
        current_soup = candidates[best_path].copy()
        current_score = candidate_scores[best_path]

        remaining_paths = [p for p in model_paths if p != best_path]

        # Greedily add models that improve performance
        print("\nGreedily adding models...")

        for iteration in range(len(remaining_paths)):
            best_addition = None
            best_addition_score = current_score

            # Try adding each remaining model
            for path in tqdm(remaining_paths, desc=f"Iteration {iteration + 1}"):
                # Create candidate soup with this model added
                candidate_soup = self._average_state_dicts(
                    selected_models + [path],
                    candidates
                )

                # Evaluate
                score = eval_metric_fn(candidate_soup, eval_dataset)

                # Check if this improves
                if score > best_addition_score:
                    best_addition = path
                    best_addition_score = score

            # Add best model if it improves
            if best_addition is not None:
                selected_models.append(best_addition)
                remaining_paths.remove(best_addition)
                current_soup = self._average_state_dicts(selected_models, candidates)
                current_score = best_addition_score

                print(f"  Added {best_addition}")
                print(f"  New score: {current_score:.4f}")
            else:
                print(f"  No improvement found. Stopping.")
                break

            # Check max models limit
            if self.max_models and len(selected_models) >= self.max_models:
                print(f"  Reached max models ({self.max_models}). Stopping.")
                break

        print(f"\nFinal soup contains {len(selected_models)} models")
        print(f"Final score: {current_score:.4f}")

        return current_soup, selected_models

    def _average_state_dicts(
        self,
        model_paths: List[str],
        loaded_state_dicts: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Average state dicts for given model paths.

        Args:
            model_paths: Paths to models to average
            loaded_state_dicts: Pre-loaded state dicts

        Returns:
            Averaged state dict
        """
        state_dicts = [loaded_state_dicts[path] for path in model_paths]

        averaged = {}
        param_keys = list(state_dicts[0].keys())

        for key in param_keys:
            params = torch.stack([sd[key] for sd in state_dicts], dim=0)
            averaged[key] = torch.mean(params, dim=0)

        return averaged

    @staticmethod
    def create_soup_from_checkpoints(
        checkpoint_dir: str,
        checkpoint_pattern: str = "checkpoint-*",
        method: str = "uniform",
        eval_dataset=None,
        eval_metric_fn: Optional[Callable] = None
    ) -> MergeResult:
        """
        Create soup from training checkpoints.

        Args:
            checkpoint_dir: Directory containing checkpoints
            checkpoint_pattern: Pattern to match checkpoint directories
            method: "uniform" or "greedy"
            eval_dataset: Evaluation dataset (for greedy)
            eval_metric_fn: Evaluation function (for greedy)

        Returns:
            MergeResult with souped model
        """
        from pathlib import Path
        from ..config import MergingConfig

        checkpoint_path = Path(checkpoint_dir)
        checkpoint_paths = sorted(checkpoint_path.glob(checkpoint_pattern))

        if not checkpoint_paths:
            raise ValueError(f"No checkpoints found matching {checkpoint_pattern} in {checkpoint_dir}")

        print(f"Found {len(checkpoint_paths)} checkpoints")

        # Create config
        config = MergingConfig(
            method="model_soups",
            model_paths=[str(p) for p in checkpoint_paths],
            soup_method=method
        )

        # Create soup
        souper = ModelSoups(config)
        return souper.merge(
            model_paths=config.model_paths,
            eval_dataset=eval_dataset,
            eval_metric_fn=eval_metric_fn
        )
