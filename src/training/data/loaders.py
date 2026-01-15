"""Data loaders for parallel corpora."""

from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path
import csv
import json
from datasets import Dataset, DatasetDict, load_dataset
import pandas as pd

from ..config import DataConfig


class ParallelCorpusLoader:
    """
    Loader for parallel translation corpora.

    Supports multiple formats:
    - TSV/CSV files
    - JSON/JSONL files
    - HuggingFace datasets
    - Separate train/eval/test files

    Features:
    - Automatic data splitting
    - Filtering and quality control
    - Data augmentation hooks
    - Caching for efficiency
    """

    def __init__(self, data_config: DataConfig):
        """
        Initialize corpus loader.

        Args:
            data_config: Data configuration
        """
        self.config = data_config

    def load(self) -> DatasetDict:
        """
        Load parallel corpus and return train/eval/test splits.

        Returns:
            DatasetDict with splits
        """
        print("\n" + "=" * 70)
        print("LOADING PARALLEL CORPUS")
        print("=" * 70)
        print(f"Data path: {self.config.data_path}")
        print(f"Format: {self.config.data_format}")
        print("=" * 70)

        # Load data based on format
        if self.config.data_format == "huggingface":
            dataset_dict = self._load_huggingface_dataset()
        elif self.config.data_format in ["tsv", "csv"]:
            dataset_dict = self._load_delimited_files()
        elif self.config.data_format in ["json", "jsonl"]:
            dataset_dict = self._load_json_files()
        else:
            raise ValueError(f"Unsupported format: {self.config.data_format}")

        # Filter data
        dataset_dict = self._filter_data(dataset_dict)

        # Print statistics
        self._print_statistics(dataset_dict)

        print("=" * 70 + "\n")

        return dataset_dict

    def _load_huggingface_dataset(self) -> DatasetDict:
        """Load HuggingFace dataset."""
        print("Loading HuggingFace dataset...")

        dataset = load_dataset(
            self.config.dataset_name,
            self.config.dataset_config,
            cache_dir=self.config.cache_dir,
        )

        # Map column names if needed
        if self.config.source_column != "source":
            dataset = dataset.rename_column(self.config.source_column, "source")
        if self.config.target_column != "target":
            dataset = dataset.rename_column(self.config.target_column, "target")

        # Convert to DatasetDict if needed
        if isinstance(dataset, Dataset):
            # Single dataset - need to split
            dataset = self._split_dataset(dataset)

        return dataset

    def _load_delimited_files(self) -> DatasetDict:
        """Load TSV/CSV files."""
        print(f"Loading {self.config.data_format.upper()} files...")

        # Check if separate files provided
        if self.config.train_file is not None:
            # Load separate files
            train_data = self._load_delimited_file(self.config.train_file)
            eval_data = self._load_delimited_file(self.config.eval_file) if self.config.eval_file else None
            test_data = self._load_delimited_file(self.config.test_file) if self.config.test_file else None

            dataset_dict = DatasetDict({
                "train": Dataset.from_dict(train_data),
            })
            if eval_data:
                dataset_dict["eval"] = Dataset.from_dict(eval_data)
            if test_data:
                dataset_dict["test"] = Dataset.from_dict(test_data)

        else:
            # Single file - split automatically
            data = self._load_delimited_file(self.config.data_path)
            dataset = Dataset.from_dict(data)
            dataset_dict = self._split_dataset(dataset)

        return dataset_dict

    def _load_delimited_file(self, file_path: str) -> Dict[str, List[str]]:
        """Load single TSV/CSV file."""
        data = {"source": [], "target": []}

        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=self.config.delimiter)

            for row in reader:
                source = row.get(self.config.source_column, "").strip()
                target = row.get(self.config.target_column, "").strip()

                data["source"].append(source)
                data["target"].append(target)

        return data

    def _load_json_files(self) -> DatasetDict:
        """Load JSON/JSONL files."""
        print(f"Loading {self.config.data_format.upper()} files...")

        if self.config.train_file is not None:
            # Load separate files
            train_data = self._load_json_file(self.config.train_file)
            eval_data = self._load_json_file(self.config.eval_file) if self.config.eval_file else None
            test_data = self._load_json_file(self.config.test_file) if self.config.test_file else None

            dataset_dict = DatasetDict({
                "train": Dataset.from_dict(train_data),
            })
            if eval_data:
                dataset_dict["eval"] = Dataset.from_dict(eval_data)
            if test_data:
                dataset_dict["test"] = Dataset.from_dict(test_data)

        else:
            # Single file - split automatically
            data = self._load_json_file(self.config.data_path)
            dataset = Dataset.from_dict(data)
            dataset_dict = self._split_dataset(dataset)

        return dataset_dict

    def _load_json_file(self, file_path: str) -> Dict[str, List[str]]:
        """Load single JSON/JSONL file."""
        data = {"source": [], "target": []}

        with open(file_path, 'r', encoding='utf-8') as f:
            if self.config.data_format == "jsonl":
                # JSONL - one JSON object per line
                for line in f:
                    obj = json.loads(line)
                    data["source"].append(obj.get(self.config.source_column, "").strip())
                    data["target"].append(obj.get(self.config.target_column, "").strip())
            else:
                # JSON - single array or object
                json_data = json.load(f)
                if isinstance(json_data, list):
                    for obj in json_data:
                        data["source"].append(obj.get(self.config.source_column, "").strip())
                        data["target"].append(obj.get(self.config.target_column, "").strip())
                else:
                    # Assume dict with lists
                    data["source"] = json_data.get(self.config.source_column, [])
                    data["target"] = json_data.get(self.config.target_column, [])

        return data

    def _split_dataset(self, dataset: Dataset) -> DatasetDict:
        """Split dataset into train/eval/test."""
        print(f"Splitting dataset with ratios: {self.config.split_ratios}")

        # Shuffle if requested
        if self.config.shuffle_before_split:
            dataset = dataset.shuffle(seed=42)

        # Calculate split sizes
        total = len(dataset)
        train_ratio, eval_ratio, test_ratio = self.config.split_ratios

        train_size = int(total * train_ratio)
        eval_size = int(total * eval_ratio)

        # Split
        splits = dataset.train_test_split(
            test_size=eval_size + (total - train_size - eval_size),
            seed=42
        )

        train_dataset = splits["train"]
        remaining = splits["test"]

        if test_ratio > 0:
            eval_test_splits = remaining.train_test_split(
                test_size=test_ratio / (eval_ratio + test_ratio),
                seed=42
            )
            eval_dataset = eval_test_splits["train"]
            test_dataset = eval_test_splits["test"]

            return DatasetDict({
                "train": train_dataset,
                "eval": eval_dataset,
                "test": test_dataset,
            })
        else:
            return DatasetDict({
                "train": train_dataset,
                "eval": remaining,
            })

    def _filter_data(self, dataset_dict: DatasetDict) -> DatasetDict:
        """Apply filtering to dataset."""
        if not any([
            self.config.remove_empty,
            self.config.remove_duplicates,
            self.config.min_source_length > 1,
            self.config.max_source_length < 512,
        ]):
            return dataset_dict

        print("Applying filters...")

        filtered_dict = DatasetDict()

        for split_name, dataset in dataset_dict.items():
            original_size = len(dataset)

            # Filter
            filtered = dataset.filter(self._filter_example)

            # Remove duplicates if requested
            if self.config.remove_duplicates:
                # Create unique key
                def add_key(example):
                    example["_key"] = example["source"] + "|" + example["target"]
                    return example

                filtered = filtered.map(add_key)
                filtered = filtered.unique("_key")
                filtered = filtered.remove_columns(["_key"])

            filtered_dict[split_name] = filtered

            removed = original_size - len(filtered)
            if removed > 0:
                print(f"  {split_name}: Removed {removed:,} samples ({removed/original_size*100:.1f}%)")

        return filtered_dict

    def _filter_example(self, example: Dict[str, Any]) -> bool:
        """Filter individual example."""
        source = example["source"]
        target = example["target"]

        # Remove empty
        if self.config.remove_empty:
            if not source or not target:
                return False

        # Length filtering
        source_len = len(source.split())
        target_len = len(target.split())

        if source_len < self.config.min_source_length or source_len > self.config.max_source_length:
            return False

        if target_len < self.config.min_target_length or target_len > self.config.max_target_length:
            return False

        # Length ratio
        if target_len > 0:
            ratio = target_len / max(source_len, 1)
            if ratio < self.config.min_length_ratio or ratio > self.config.max_length_ratio:
                return False

        return True

    def _print_statistics(self, dataset_dict: DatasetDict):
        """Print dataset statistics."""
        print("\nDataset Statistics:")
        print("-" * 50)

        for split_name, dataset in dataset_dict.items():
            print(f"{split_name.capitalize():10} : {len(dataset):,} samples")

        print("-" * 50)

    @staticmethod
    def save_to_tsv(
        dataset: Dataset,
        output_path: str,
        source_column: str = "source",
        target_column: str = "target"
    ):
        """
        Save dataset to TSV file.

        Args:
            dataset: Dataset to save
            output_path: Output file path
            source_column: Source column name
            target_column: Target column name
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[source_column, target_column], delimiter='\t')
            writer.writeheader()

            for example in dataset:
                writer.writerow({
                    source_column: example["source"],
                    target_column: example["target"],
                })

        print(f"✓ Saved {len(dataset):,} samples to {output_path}")
