"""Dataset Deduplication."""

import hashlib
from typing import Dict, List, Set

from AuraTrainer.utils.logger import get_logger

logger = get_logger("AuraTrainer.Deduplicator")


class DatasetDeduplicator:
    """Remove duplicate examples from datasets."""

    def __init__(self, method: str = "exact"):
        """Initialize deduplicator.

        Args:
            method: Deduplication method ('exact' or 'hash').
        """
        self.method = method
        self._seen_hashes: Set[str] = set()
        self._seen_texts: Set[str] = set()

    def _compute_hash(self, text: str) -> str:
        """Compute MD5 hash of text.

        Args:
            text: Input text.

        Returns:
            Hex digest of the hash.
        """
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _normalize_for_dedup(self, text: str) -> str:
        """Normalize text for deduplication comparison.

        Args:
            text: Input text.

        Returns:
            Normalized text.
        """
        text = text.lower().strip()
        text = " ".join(text.split())
        return text

    def is_duplicate(self, example: Dict[str, str], key_field: str = "output") -> bool:
        """Check if an example is a duplicate.

        Args:
            example: Example to check.
            key_field: Field to use for deduplication.

        Returns:
            True if duplicate.
        """
        text = example.get(key_field, "")

        if not text.strip():
            return True

        if self.method == "hash":
            text_hash = self._compute_hash(text)
            if text_hash in self._seen_hashes:
                return True
            self._seen_hashes.add(text_hash)
        else:
            normalized = self._normalize_for_dedup(text)
            if normalized in self._seen_texts:
                return True
            self._seen_texts.add(normalized)

        return False

    def deduplicate_batch(
        self, examples: List[Dict[str, str]], key_field: str = "output"
    ) -> List[Dict[str, str]]:
        """Remove duplicates from a batch.

        Args:
            examples: List of examples.
            key_field: Field to use for deduplication.

        Returns:
            List of unique examples.
        """
        unique = []
        duplicates = 0

        for example in examples:
            if not self.is_duplicate(example, key_field):
                unique.append(example)
            else:
                duplicates += 1

        if duplicates > 0:
            logger.info(
                f"Deduplication: removed {duplicates} duplicates "
                f"({len(unique)} unique from {len(examples)} total)"
            )

        return unique

    def deduplicate_across_datasets(
        self, datasets: Dict[str, List[Dict[str, str]]], key_field: str = "output"
    ) -> Dict[str, List[Dict[str, str]]]:
        """Deduplicate across multiple datasets.

        Args:
            datasets: Dictionary mapping dataset names to example lists.
            key_field: Field to use for deduplication.

        Returns:
            Dictionary with deduplicated datasets.
        """
        self._seen_hashes.clear()
        self._seen_texts.clear()

        result = {}
        total_before = 0
        total_after = 0

        for name, examples in datasets.items():
            total_before += len(examples)
            deduped = self.deduplicate_batch(examples, key_field)
            result[name] = deduped
            total_after += len(deduped)

        removed = total_before - total_after
        if removed > 0:
            logger.info(
                f"Cross-dataset dedup: removed {removed} duplicates "
                f"({total_after} unique from {total_before} total)"
            )

        return result

    def get_stats(self) -> Dict[str, int]:
        """Get deduplication statistics.

        Returns:
            Dictionary with dedup stats.
        """
        return {
            "seen_hashes": len(self._seen_hashes),
            "seen_texts": len(self._seen_texts),
            "method": self.method,
        }

    def reset(self) -> None:
        """Reset deduplication state."""
        self._seen_hashes.clear()
        self._seen_texts.clear()
