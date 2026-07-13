"""Dataset Formatting for TinyLlama Chat Format."""

from typing import Dict, List, Optional

from AuraTrainer.utils.logger import get_logger

logger = get_logger("AuraTrainer.DatasetFormatter")


# System prompt for the Aura Book assistant
AURA_BOOK_SYSTEM_PROMPT = (
    "You are Aura, a helpful university assistant specializing in programming, "
    "mathematics, and education. You can explain programming concepts, write code, "
    "debug errors, explain math problems, and help with studying. You speak both "
    "Arabic and English fluently."
)

CODE_INSTRUCTION_TEMPLATE = (
    "<|system|>\n{system}</s>\n"
    "<|user|>\n{instruction}\n{input}</s>\n"
    "<|assistant|>\n{output}</s>"
)

CHAT_TEMPLATE = (
    "<|system|>\n{system}</s>\n"
    "<|user|>\n{user_message}</s>\n"
    "<|assistant|>\n{assistant_message}</s>"
)

MATH_TEMPLATE = (
    "<|system|>\n{system}</s>\n"
    "<|user|>\nSolve the following math problem step by step:\n{question}</s>\n"
    "<|assistant|>\n{answer}</s>"
)

CODE_ONLY_TEMPLATE = (
    "<|system|>\n{system}</s>\n"
    "<|user|>\n{question}</s>\n"
    "<|assistant|>\n{answer}</s>"
)


class DatasetFormatter:
    """Format datasets into TinyLlama chat format."""

    def __init__(self, system_prompt: Optional[str] = None):
        """Initialize formatter.

        Args:
            system_prompt: Custom system prompt. Uses Aura Book default if None.
        """
        self.system_prompt = system_prompt or AURA_BOOK_SYSTEM_PROMPT

    def format_code_instruction(self, example: Dict[str, str]) -> str:
        """Format code instruction example.

        Args:
            example: Dict with instruction, input, output fields.

        Returns:
            Formatted string.
        """
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output = example.get("output", "")

        if input_text:
            user_msg = f"{instruction}\n\nInput:\n{input_text}"
        else:
            user_msg = instruction

        return CODE_INSTRUCTION_TEMPLATE.format(
            system=self.system_prompt,
            instruction=user_msg,
            output=output,
        )

    def format_chat(self, example: Dict[str, str]) -> str:
        """Format chat conversation example.

        Args:
            example: Dict with user/assistant messages.

        Returns:
            Formatted string.
        """
        user_msg = example.get("prompt", example.get("instruction", ""))
        assistant_msg = example.get("response", example.get("output", ""))

        return CHAT_TEMPLATE.format(
            system=self.system_prompt,
            user_message=user_msg,
            assistant_message=assistant_msg,
        )

    def format_math(self, example: Dict[str, str]) -> str:
        """Format math problem example.

        Args:
            example: Dict with question and answer.

        Returns:
            Formatted string.
        """
        question = example.get("question", example.get("query", ""))
        answer = example.get("answer", example.get("solution", ""))

        return MATH_TEMPLATE.format(
            system=self.system_prompt,
            question=question,
            answer=answer,
        )

    def format_education(self, example: Dict[str, str]) -> str:
        """Format education example.

        Args:
            example: Dict with input and output.

        Returns:
            Formatted string.
        """
        input_text = example.get("input", example.get("text", ""))
        output = example.get("output", "")

        return CODE_ONLY_TEMPLATE.format(
            system=self.system_prompt,
            question=input_text,
            answer=output,
        )

    def format_arabic(self, example: Dict[str, str]) -> str:
        """Format Arabic language example.

        Args:
            example: Dict with instruction/input/output.

        Returns:
            Formatted string.
        """
        instruction = example.get("instruction", example.get("input", ""))
        input_text = example.get("input", "")
        output = example.get("output", "")

        if input_text and input_text != instruction:
            user_msg = f"{instruction}\n\n{input_text}"
        else:
            user_msg = instruction

        return CODE_INSTRUCTION_TEMPLATE.format(
            system=self.system_prompt,
            instruction=user_msg,
            output=output,
        )

    def format_by_category(
        self, example: Dict[str, str], category: str
    ) -> str:
        """Format example based on its category.

        Args:
            example: Example to format.
            category: Dataset category.

        Returns:
            Formatted string.
        """
        formatters = {
            "programming": self.format_code_instruction,
            "chat": self.format_chat,
            "math": self.format_math,
            "education": self.format_education,
            "arabic": self.format_arabic,
        }

        formatter = formatters.get(category, self.format_chat)
        return formatter(example)

    def format_batch(
        self, examples: List[Dict[str, str]], category: str
    ) -> List[str]:
        """Format a batch of examples.

        Args:
            examples: List of examples to format.
            category: Dataset category.

        Returns:
            List of formatted strings.
        """
        return [self.format_by_category(ex, category) for ex in examples]

    def tokenize_formatted(
        self, text: str, tokenizer, max_length: int = 2048
    ) -> Dict[str, List[int]]:
        """Tokenize a formatted text string.

        Args:
            text: Formatted text.
            tokenizer: Tokenizer instance.
            max_length: Maximum sequence length.

        Returns:
            Dictionary with input_ids and attention_mask.
        """
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )
        return encoded
