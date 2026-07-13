"""Dataset Sampling and Balancing."""

import random
from typing import Any, Dict, List, Optional

from AuraTrainer.utils.logger import get_logger

logger = get_logger("AuraTrainer.Sampler")


# Default category ratios
DEFAULT_RATIOS = {
    "programming": 0.40,
    "chat": 0.25,
    "math": 0.15,
    "education": 0.10,
    "arabic": 0.10,
}

# Dataset size presets
SIZE_PRESETS = {
    "50k": 50_000,
    "100k": 100_000,
    "250k": 250_000,
    "500k": 500_000,
    "1M": 1_000_000,
    "2M": 2_000_000,
}


class DatasetSampler:
    """Sample and balance datasets according to configured ratios."""

    def __init__(
        self,
        ratios: Optional[Dict[str, float]] = None,
        seed: int = 42,
        shuffle: bool = True,
    ):
        """Initialize sampler.

        Args:
            ratios: Category sampling ratios. Uses defaults if None.
            seed: Random seed for reproducibility.
            shuffle: Whether to shuffle sampled data.
        """
        self.ratios = ratios or DEFAULT_RATIOS
        self.seed = seed
        self.shuffle = shuffle
        self._rng = random.Random(seed)

        total = sum(self.ratios.values())
        if abs(total - 1.0) > 0.001:
            logger.warning(
                f"Ratios sum to {total:.3f}, normalizing to 1.0"
            )
            self.ratios = {k: v / total for k, v in self.ratios.items()}

    def calculate_counts(
        self, total_samples: int, available: Optional[Dict[str, int]] = None
    ) -> Dict[str, int]:
        """Calculate how many samples to take from each category.

        Args:
            total_samples: Total number of samples desired.
            available: Available samples per category.

        Returns:
            Dictionary mapping category to sample count.
        """
        counts = {}
        for category, ratio in self.ratios.items():
            desired = int(total_samples * ratio)
            if available and category in available:
                counts[category] = min(desired, available[category])
            else:
                counts[category] = desired

        actual_total = sum(counts.values())
        if actual_total < total_samples:
            deficit = total_samples - actual_total
            for cat in counts:
                counts[cat] += int(deficit * self.ratios.get(cat, 0))

        return counts

    def sample_dataset(
        self,
        dataset: List[Dict[str, str]],
        num_samples: int,
        category: str,
    ) -> List[Dict[str, str]]:
        """Sample from a single dataset.

        Args:
            dataset: Full dataset list.
            num_samples: Number of samples to take.
            category: Dataset category for logging.

        Returns:
            Sampled subset.
        """
        if num_samples >= len(dataset):
            if self.shuffle:
                sampled = list(dataset)
                self._rng.shuffle(sampled)
                return sampled
            return list(dataset)

        indices = self._rng.sample(range(len(dataset)), num_samples)
        sampled = [dataset[i] for i in sorted(indices)]

        logger.info(
            f"Sampled {len(sampled)} from {category} "
            f"(requested: {num_samples}, available: {len(dataset)})"
        )
        return sampled

    def balance_datasets(
        self,
        datasets: Dict[str, List[Dict[str, str]]],
        total_samples: int,
        dataset_size: str = "100k",
    ) -> List[Dict[str, str]]:
        """Balance and merge datasets according to ratios.

        Args:
            datasets: Dictionary mapping category to dataset list.
            total_samples: Total desired samples.
            dataset_size: Size preset name.

        Returns:
            Balanced and merged dataset list.
        """
        if dataset_size in SIZE_PRESETS:
            total_samples = SIZE_PRESETS[dataset_size]

        available = {cat: len(ds) for cat, ds in datasets.items()}
        counts = self.calculate_counts(total_samples, available)

        all_samples = []
        for category, ds in datasets.items():
            count = counts.get(category, 0)
            if count <= 0:
                continue
            sampled = self.sample_dataset(ds, count, category)
            for s in sampled:
                s["_category"] = category
            all_samples.extend(sampled)

        if self.shuffle:
            self._rng.shuffle(all_samples)

        logger.info(
            f"Balanced dataset: {len(all_samples)} total samples "
            f"from {len(datasets)} categories"
        )
        for cat, count in counts.items():
            logger.info(f"  {cat}: {count} samples")

        return all_samples

    def get_size_bytes(self, dataset: List[Dict[str, str]]) -> int:
        """Estimate dataset size in bytes.

        Args:
            dataset: Dataset list.

        Returns:
            Estimated size in bytes.
        """
        total = 0
        for example in dataset:
            for value in example.values():
                if isinstance(value, str):
                    total += len(value.encode("utf-8"))
        return total

    def split_dataset(
        self,
        dataset: List[Dict[str, str]],
        train_ratio: float = 0.9,
        eval_ratio: float = 0.1,
    ) -> Dict[str, List[Dict[str, str]]]:
        """Split dataset into train and eval sets.

        Args:
            dataset: Full dataset.
            train_ratio: Fraction for training.
            eval_ratio: Fraction for evaluation.

        Returns:
            Dictionary with 'train' and 'eval' splits.
        """
        if self.shuffle:
            shuffled = list(dataset)
            self._rng.shuffle(shuffled)
        else:
            shuffled = dataset

        split_idx = int(len(shuffled) * train_ratio)
        return {
            "train": shuffled[:split_idx],
            "eval": shuffled[split_idx:],
        }
