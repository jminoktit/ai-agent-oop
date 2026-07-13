"""Dataset Loading and Streaming Support."""

from typing import Any, Dict, Generator, List, Optional

from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)

from AuraTrainer.utils.logger import get_logger

logger = get_logger("AuraTrainer.DatasetLoader")


class DatasetLoader:
    """Load datasets from HuggingFace with streaming support."""

    def __init__(self, streaming: bool = True):
        """Initialize loader.

        Args:
            streaming: Whether to use streaming mode.
        """
        self.streaming = streaming
        self._loaded_datasets: Dict[str, Any] = {}

    def load_single(
        self,
        name: str,
        repo: str,
        subset: Optional[str] = None,
        split: str = "train",
        streaming: Optional[bool] = None,
    ) -> Any:
        """Load a single dataset.

        Args:
            name: Dataset identifier.
            repo: HuggingFace repo path.
            subset: Dataset subset/config.
            split: Dataset split.
            streaming: Override streaming mode.

        Returns:
            Loaded dataset (IterableDataset or Dataset).
        """
        use_streaming = streaming if streaming is not None else self.streaming

        cache_key = f"{repo}:{subset}:{split}:{use_streaming}"
        if cache_key in self._loaded_datasets:
            logger.info(f"Using cached dataset: {name}")
            return self._loaded_datasets[cache_key]

        try:
            logger.info(f"Loading dataset: {name} from {repo} (stream={use_streaming})")
            ds = load_dataset(
                repo,
                subset,
                split=split,
                streaming=use_streaming,
                trust_remote_code=True,
            )
            self._loaded_datasets[cache_key] = ds
            return ds
        except Exception as e:
            logger.error(f"Failed to load dataset {name}: {e}")
            raise

    def load_multiple(
        self, configs: List[Dict[str, Any]], streaming: Optional[bool] = None
    ) -> Dict[str, Any]:
        """Load multiple datasets.

        Args:
            configs: List of dataset configuration dicts.
            streaming: Override streaming mode.

        Returns:
            Dictionary mapping dataset names to loaded datasets.
        """
        datasets = {}
        for cfg in configs:
            try:
                ds = self.load_single(
                    name=cfg["name"],
                    repo=cfg["repo"],
                    subset=cfg.get("subset"),
                    split=cfg.get("split", "train"),
                    streaming=streaming,
                )
                datasets[cfg["name"]] = ds
            except Exception as e:
                logger.warning(f"Skipping dataset {cfg.get('name', 'unknown')}: {e}")
        return datasets

    def iterate_dataset(
        self, dataset: Any, fields: Dict[str, str]
    ) -> Generator[Dict[str, str], None, None]:
        """Iterate over a dataset yielding standardized examples.

        Args:
            dataset: The dataset to iterate.
            fields: Mapping of standardized field names to source field names.

        Yields:
            Standardized example dictionaries.
        """
        if isinstance(dataset, IterableDataset):
            for example in dataset:
                yield self._extract_fields(example, fields)
        else:
            for example in dataset:
                yield self._extract_fields(example, fields)

    def _extract_fields(
        self, example: Dict[str, Any], fields: Dict[str, str]
    ) -> Dict[str, str]:
        """Extract and standardize fields from an example.

        Args:
            example: Raw example from dataset.
            fields: Mapping of output names to source field names.

        Returns:
            Extracted fields dictionary.
        """
        result = {}
        for target_name, source_name in fields.items():
            value = example.get(source_name, "")
            if isinstance(value, list):
                value = str(value)
            elif value is None:
                value = ""
            result[target_name] = str(value)
        return result

    def get_dataset_info(self, name: str, repo: str, subset: Optional[str] = None) -> dict:
        """Get dataset info without loading it.

        Args:
            name: Dataset identifier.
            repo: HuggingFace repo path.
            subset: Dataset subset/config.

        Returns:
            Dictionary with dataset info.
        """
        try:
            from huggingface_hub import dataset_info

            info = dataset_info(repo, subset)
            return {
                "name": name,
                "repo": repo,
                "subset": subset,
                "description": getattr(info, "description", ""),
                "size_bytes": getattr(info, "size_bytes", 0),
                "splits": list(getattr(info, "splits", {}).keys()),
            }
        except Exception as e:
            logger.warning(f"Could not get info for {name}: {e}")
            return {"name": name, "repo": repo}

    def clear_cache(self) -> None:
        """Clear cached datasets."""
        self._loaded_datasets.clear()
        logger.info("Dataset cache cleared")
