"""Dataset Cleaning Utilities."""

import re
from typing import Dict, List, Optional

from AuraTrainer.utils.logger import get_logger

logger = get_logger("AuraTrainer.DatasetCleaner")


class DatasetCleaner:
    """Clean and validate dataset examples."""

    def __init__(
        self,
        min_text_length: int = 10,
        max_text_length: int = 8192,
        min_words: int = 3,
        max_words: int = 4096,
        remove_empty: bool = True,
        remove_broken_code: bool = True,
    ):
        """Initialize cleaner.

        Args:
            min_text_length: Minimum character length.
            max_text_length: Maximum character length.
            min_words: Minimum word count.
            max_words: Maximum word count.
            remove_empty: Remove empty examples.
            remove_broken_code: Detect and remove broken code.
        """
        self.min_text_length = min_text_length
        self.max_text_length = max_text_length
        self.min_words = min_words
        self.max_words = max_words
        self.remove_empty = remove_empty
        self.remove_broken_code = remove_broken_code

        self._broken_patterns = [
            re.compile(r"^\s*(def|class|import|from)\s*$"),
            re.compile(r"^\s*(if|else|elif|for|while|try|except)\s*$"),
            re.compile(r"^\s*(return|yield|raise|assert)\s*$"),
            re.compile(r"[^{}\[\]()]*$"),
            re.compile(r"\bdef\s+\w+\s*\([^)]*$"),
            re.compile(r"\bclass\s+\w+\s*\([^)]*$"),
        ]

    def is_valid(self, example: Dict[str, str], text_field: str = "output") -> bool:
        """Check if an example passes all cleaning rules.

        Args:
            example: The example to validate.
            text_field: Which field to check for text quality.

        Returns:
            True if example is valid.
        """
        text = example.get(text_field, "")

        if self.remove_empty and not text.strip():
            return False

        if len(text) < self.min_text_length:
            return False

        if len(text) > self.max_text_length:
            return False

        words = text.split()
        if len(words) < self.min_words:
            return False
        if len(words) > self.max_words:
            return False

        if self.remove_broken_code and self._is_broken_code(text):
            return False

        return True

    def clean_example(
        self, example: Dict[str, str], text_field: str = "output"
    ) -> Optional[Dict[str, str]]:
        """Clean a single example by trimming and normalizing.

        Args:
            example: The example to clean.
            text_field: Main text field name.

        Returns:
            Cleaned example or None if invalid.
        """
        cleaned = {}

        for key, value in example.items():
            if isinstance(value, str):
                cleaned[key] = self._normalize_text(value)
            else:
                cleaned[key] = value

        if not self.is_valid(cleaned, text_field):
            return None

        return cleaned

    def clean_batch(
        self, examples: List[Dict[str, str]], text_field: str = "output"
    ) -> List[Dict[str, str]]:
        """Clean a batch of examples.

        Args:
            examples: List of examples to clean.
            text_field: Main text field name.

        Returns:
            List of valid cleaned examples.
        """
        cleaned = []
        for ex in examples:
            result = self.clean_example(ex, text_field)
            if result is not None:
                cleaned.append(result)

        removed = len(examples) - len(cleaned)
        if removed > 0:
            logger.info(f"Cleaned: removed {removed}/{len(examples)} examples")

        return cleaned

    def _normalize_text(self, text: str) -> str:
        """Normalize text by cleaning whitespace and control characters.

        Args:
            text: Input text.

        Returns:
            Normalized text.
        """
        text = text.strip()
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)
        return text

    def _is_broken_code(self, text: str) -> bool:
        """Detect broken or incomplete code.

        Args:
            text: Code text to check.

        Returns:
            True if code appears broken.
        """
        text = text.strip()

        open_parens = text.count("(") - text.count(")")
        if open_parens > 2:
            return True

        open_brackets = text.count("[") - text.count("]")
        if open_brackets > 2:
            return True

        open_braces = text.count("{") - text.count("}")
        if open_braces > 2:
            return True

        lines = text.split("\n")
        if lines:
            last_line = lines[-1].strip()
            for pattern in self._broken_patterns:
                if pattern.match(last_line) and len(lines) > 1:
                    return True

        return False
