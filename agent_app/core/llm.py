import json
import os
import urllib.request
from typing import Dict, List


class OllamaLLM:
    def __init__(self, model: str = None, base_url: str = None, temperature: float = 0.7, timeout: int = None):
        self.model = model or os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:14b")
        self.base_url = (base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.temperature = temperature
        self.timeout = timeout
        self._detect_model()

    def _detect_model(self) -> None:
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
            models = [m["name"] for m in data.get("models", [])]
            if self.model not in models and models:
                self.model = models[0]
        except Exception:
            pass

    def generate(self, messages: List[Dict[str, str]], system_prompt: str = "") -> str:
        prompt = self._build_prompt(messages, system_prompt)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                f"{self.base_url}/api/generate",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                response_text = resp.read().decode()

            output = []
            for line in response_text.strip().split("\n"):
                if line.strip():
                    chunk = json.loads(line)
                    output.append(chunk.get("response", ""))
                    if chunk.get("done", False):
                        break
            return "".join(output)
        except Exception as e:
            return f"LLM Error: {e}"

    def _build_prompt(self, messages: List[Dict[str, str]], system_prompt: str) -> str:
        parts = []
        if system_prompt:
            parts.append(f"System: {system_prompt}")
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prefix = "Assistant" if role == "assistant" else "User"
            parts.append(f"{prefix}: {content}")
        parts.append("Assistant:")
        return "\n".join(parts)

    def update_model(self, model: str) -> None:
        self.model = model

    def __repr__(self) -> str:
        return f"OllamaLLM(model={self.model!r})"
