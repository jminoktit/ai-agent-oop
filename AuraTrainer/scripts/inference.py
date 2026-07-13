#!/usr/bin/env python3
"""Inference and Testing Script.

Test the trained AuraBook assistant on various domains.

Usage:
    python -m AuraTrainer.scripts.inference \
        --model-path ./outputs/merged \
        --interactive

    python -m AuraTrainer.scripts.inference \
        --model-path ./outputs/lora_adapter \
        --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --test-all
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from AuraTrainer.utils.logger import setup_logger, get_logger

logger = get_logger("AuraTrainer.Inference")

# ─── Test Prompts ───────────────────────────────────────

TEST_PROMPTS = {
    "Python": [
        "Write a Python function to find the longest common subsequence of two strings.",
        "Explain the difference between a list and a tuple in Python.",
        "Write a Python class implementing a binary search tree.",
    ],
    "Django": [
        "How do I create a Django REST API endpoint for user registration?",
        "Explain Django middleware and how to create a custom one.",
        "Write a Django model for a blog post with comments.",
    ],
    "SQL": [
        "Write a SQL query to find the second highest salary in an employees table.",
        "Explain the difference between INNER JOIN and LEFT JOIN.",
        "Write a SQL query to find duplicate records in a table.",
    ],
    "Algorithms": [
        "Explain the time complexity of quicksort and when it performs worst.",
        "Implement a breadth-first search algorithm in Python.",
        "What is dynamic programming? Give an example with the knapsack problem.",
    ],
    "Math": [
        "Explain the chain rule in calculus with an example.",
        "Solve: What is the integral of x^2 * e^x dx?",
        "Explain eigenvalues and eigenvectors in linear algebra.",
    ],
    "Arabic": [
        "اشرح لي كيف تعمل هياكل البيانات في البرمجة.",
        "اكتب لي دالة في بايثون تتحقق من إذا كان النص palindrome.",
        "ما هو الفرق بين AlGORITHM و DATA STRUCTURE؟ اشرح بالعربي.",
    ],
    "English": [
        "Explain the SOLID principles in software engineering.",
        "What is the difference between REST and GraphQL APIs?",
        "Explain how garbage collection works in Python.",
    ],
}


class AuraInference:
    """Run inference on the trained model."""

    def __init__(
        self,
        model_path: str,
        base_model: Optional[str] = None,
        device: str = "auto",
    ):
        """Initialize inference engine.

        Args:
            model_path: Path to model or adapter.
            base_model: Base model name if loading adapter.
            device: Device to use.
        """
        self.model_path = model_path
        self.base_model = base_model
        self.device = device
        self.pipe = None

        self._load_model()

    def _load_model(self) -> None:
        """Load model and create pipeline."""
        logger.info(f"Loading model from {self.model_path}")

        if self.base_model:
            logger.info(f"Base model: {self.base_model}")
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                torch_dtype=torch.float16,
                device_map=self.device,
                trust_remote_code=True,
            )
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, self.model_path)
            model = model.merge_and_unload()
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map=self.device,
                trust_remote_code=True,
            )

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path if not self.base_model else self.base_model,
            trust_remote_code=True,
        )

        self.pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,
            device_map=self.device,
        )

        logger.info("Model loaded successfully!")

    def generate(
        self,
        prompt: str,
        system_prompt: str = "You are Aura, a helpful university assistant.",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        """Generate a response.

        Args:
            prompt: User prompt.
            system_prompt: System prompt.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Top-p sampling.
            do_sample: Whether to sample.

        Returns:
            Generated response text.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        formatted = self.pipe.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        outputs = self.pipe(
            formatted,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            repetition_penalty=1.1,
        )

        response = outputs[0]["generated_text"]
        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1].strip()
        if "</s>" in response:
            response = response.replace("</s>", "").strip()

        return response

    def test_category(
        self, category: str, prompts: List[str], verbose: bool = True
    ) -> List[dict]:
        """Test all prompts in a category.

        Args:
            category: Category name.
            prompts: List of test prompts.
            verbose: Print results.

        Returns:
            List of result dictionaries.
        """
        results = []

        if verbose:
            print(f"\n{'='*60}")
            print(f"  Testing: {category}")
            print(f"{'='*60}")

        for i, prompt in enumerate(prompts, 1):
            if verbose:
                print(f"\n--- {category} Test {i} ---")
                print(f"Prompt: {prompt}")
                print(f"\nResponse:")

            response = self.generate(prompt)

            if verbose:
                print(response)
                print()

            results.append({
                "category": category,
                "prompt": prompt,
                "response": response,
            })

        return results

    def test_all(self, verbose: bool = True) -> dict:
        """Run all test categories.

        Args:
            verbose: Print results.

        Returns:
            Dictionary of all results.
        """
        all_results = {}

        for category, prompts in TEST_PROMPTS.items():
            results = self.test_category(category, prompts, verbose)
            all_results[category] = results

        if verbose:
            print(f"\n{'='*60}")
            print("  All tests completed!")
            print(f"{'='*60}")
            total = sum(len(r) for r in all_results.values())
            print(f"Total tests: {total}")

        return all_results

    def interactive_mode(self) -> None:
        """Run interactive chat mode."""
        print("\n" + "=" * 60)
        print("  AuraBook Assistant - Interactive Mode")
        print("  Type 'quit' to exit, 'test' to run all tests")
        print("=" * 60 + "\n")

        system_prompt = (
            "You are Aura, a helpful university assistant specializing in "
            "programming, mathematics, and education. You can explain "
            "programming concepts, write code, debug errors, explain math "
            "problems, and help with studying. You speak both Arabic and "
            "English fluently."
        )

        while True:
            try:
                user_input = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            if user_input.lower() == "test":
                self.test_all()
                continue

            response = self.generate(user_input, system_prompt)
            print(f"\nAura: {response}")


def main():
    """Main inference entry point."""
    parser = argparse.ArgumentParser(description="AuraBook Inference")
    parser.add_argument(
        "--model-path", required=True, help="Path to model or adapter"
    )
    parser.add_argument("--base-model", default=None, help="Base model if using adapter")
    parser.add_argument("--device", default="auto", help="Device to use")
    parser.add_argument("--test-all", action="store_true", help="Run all tests")
    parser.add_argument("--test-category", default=None, help="Test specific category")
    parser.add_argument("--interactive", action="store_true", help="Interactive chat")
    parser.add_argument("--prompt", default=None, help="Single prompt to test")
    args = parser.parse_args()

    setup_logger("AuraTrainer")

    engine = AuraInference(
        model_path=args.model_path,
        base_model=args.base_model,
        device=args.device,
    )

    if args.interactive:
        engine.interactive_mode()
    elif args.test_all:
        engine.test_all()
    elif args.test_category:
        prompts = TEST_PROMPTS.get(args.test_category, [])
        if prompts:
            engine.test_category(args.test_category, prompts)
        else:
            print(f"Unknown category: {args.test_category}")
            print(f"Available: {', '.join(TEST_PROMPTS.keys())}")
    elif args.prompt:
        response = engine.generate(args.prompt)
        print(f"\nAura: {response}")
    else:
        print("Specify --interactive, --test-all, --test-category, or --prompt")
        print("Example: python -m AuraTrainer.scripts.inference --model-path ./outputs/merged --test-all")


if __name__ == "__main__":
    main()
