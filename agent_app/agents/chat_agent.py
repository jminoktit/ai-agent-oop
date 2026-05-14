from typing import Optional

from ..core import BaseAgent, OllamaLLM
from ..tools import KeyboardConverter, RuleEngine


class ChatAgent(BaseAgent):
    def __init__(
        self,
        name: str = "ChatBot",
        system_prompt: str = (
            "You are a friendly and helpful AI assistant. "
            "IMPORTANT: If the user's message does not make sense, "
            "is gibberish, or appears to be typed with the wrong "
            "keyboard layout, reply ONLY with: "
            "I don't understand. "
            "Do not guess or try to interpret gibberish."
        ),
        llm: Optional[OllamaLLM] = None,
    ):
        super().__init__(name, system_prompt)
        self.llm = llm or OllamaLLM()
        self.rule_engine = RuleEngine()
        self.keyboard_conv = KeyboardConverter()

    def run(self, input_data: str) -> str:
        rule_result = self.rule_engine.check(input_data, "chat")
        if rule_result is not None:
            if input_data.lower().startswith(("learn:", "learn ", "rules:")):
                return rule_result
            self.add_memory("user", input_data)
            self.add_memory("tool", f"[rule] {rule_result}")
            return rule_result

        self.add_memory("user", input_data)

        tool_result = self._check_tools(input_data)
        if tool_result is not None:
            self.add_memory("tool", tool_result)
            response = self._ask_llm()
        else:
            response = self._ask_llm()

        # if LLM returned error or gibberish, retry with keyboard conversion (invisible)
        if self._looks_like_failure(response):
            converted = self._auto_convert_keyboard(input_data)
            if converted:
                retry = self._ask_llm_with(converted)
                if not self._looks_like_failure(retry):
                    response = retry

        self.add_memory("assistant", response)
        return response

    def _looks_like_failure(self, text: str) -> bool:
        lower = text.lower()
        fails = [
            "llm error", "i don't understand", "i do not understand",
            "i'm not sure", "i am not sure", "could you rephrase",
            "can you rephrase", "doesn't make sense", "not make sense",
            "i didn't understand", "i did not understand",
            "could you clarify", "can you clarify",
        ]
        if any(f in lower for f in fails):
            return True
        return False

    def _auto_convert_keyboard(self, text: str) -> Optional[str]:
        if len(text) < 2:
            return None

        common_en_words = {"the", "is", "it", "to", "in", "and", "of", "a", "an", "this", "that",
                           "for", "on", "with", "as", "at", "by", "from", "or", "be", "are", "was",
                           "have", "has", "had", "do", "does", "did", "will", "would", "can", "could",
                           "shall", "should", "may", "might", "must", "not", "no", "yes", "hello",
                           "hi", "what", "how", "why", "when", "where", "who", "which", "please",
                           "thanks", "thank", "help", "me", "my", "i", "we", "they", "he", "she",
                           "am", "is", "are", "was", "were", "been", "being", "have", "has", "had",
                           "do", "does", "did", "will", "would", "can", "could", "shall", "should",
                           "may", "might", "must", "need", "want", "like", "love", "good", "bad",
                           "big", "small", "new", "old", "up", "down", "go", "get", "make", "take",
                           "know", "see", "look", "find", "give", "tell", "ask", "use", "work",
                           "run", "set", "put", "try", "keep", "let", "start", "stop", "done"}
        words = text.lower().split()
        en_word_count = sum(1 for w in words if w in common_en_words)
        ar_chars = sum(1 for c in text if "\u0600" <= c <= "\u06FF")
        total_letters = sum(1 for c in text if c.isalpha())
        has_numbers = any(c.isdigit() for c in text)
        ar_ratio = ar_chars / total_letters if total_letters > 0 else 0

        # Case 1: English keyboard typed when meaning Arabic (e.g. "lklh" -> "منما")
        if ar_chars == 0 and en_word_count == 0 and not has_numbers and len(text) > 1:
            converted = self.keyboard_conv.convert(text)
            if "→" in converted:
                clean = converted.split("→")[-1].strip()
                if clean and clean != text:
                    return clean

        # Case 2: Arabic keyboard typed when meaning English (e.g. "هيثتف" -> "hello")
        if ar_ratio > 0.5:
            converted = self.keyboard_conv.convert(text)
            if "→" in converted:
                en_guess = converted.split("→")[-1].strip()
                if en_guess and en_guess != text:
                    has_vowel = any(v in en_guess.lower() for v in ("a", "e", "i", "o", "u"))
                    en_letters = sum(1 for c in en_guess if c.isascii() and c.isalpha())
                    if has_vowel and en_letters >= len(text) * 0.5:
                        return en_guess

        return None

    def _check_tools(self, input_data: str) -> Optional[str]:
        for tool in self.tools.values():
            if tool.name in input_data.lower() or input_data.startswith(f"{tool.name}:"):
                return tool.execute(input_data)
        return None

    def _ask_llm(self) -> str:
        history = self.get_memory(recent=20)
        return self.llm.generate(history, self.system_prompt)

    def _ask_llm_with(self, text: str) -> str:
        history = self.get_memory(recent=20)
        for i in range(len(history) - 1, -1, -1):
            if history[i]["role"] == "user":
                history[i]["content"] = text
                break
        return self.llm.generate(history, self.system_prompt)

    def reset(self) -> None:
        self._clear_memory()
        self.stop()
