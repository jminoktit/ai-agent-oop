import re
from typing import Optional

from ..core import BaseTool


EN_TO_AR = {
    "q": "ض", "w": "ص", "e": "ث", "r": "ق", "t": "ف", "y": "غ",
    "u": "ع", "i": "ه", "o": "خ", "p": "ح", "[": "ج", "]": "د",
    "a": "ش", "s": "س", "d": "ي", "f": "ب", "g": "ل", "h": "ا",
    "j": "ت", "k": "ن", "l": "م", ";": "ك", "'": "ط",
    "z": "ئ", "x": "ء", "c": "ؤ", "v": "ر", "b": "لا", "n": "ى",
    "m": "ة", ",": "و", ".": "ز", "/": "ظ",
    "Q": "َ", "W": "ً", "E": "ُ", "R": "ٌ", "T": "لإ",
    "Y": "إ", "U": "`", "I": "÷", "O": "×", "P": "؛",
    "A": "ِ", "S": "ٍ", "D": "]", "F": "[", "G": "أ",
    "H": "ـ", "J": "،", "K": "/", "L": ":", ":": '"',
    "Z": "~", "X": "ْ", "C": "}", "V": "{", "B": "لآ",
    "N": "آ", "M": "'", "<": ",", ">": ".", "?": "؟",
}

AR_TO_EN = {v: k for k, v in EN_TO_AR.items()}
AR_TO_EN[" "] = " "


def is_arabic(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", text))


def has_arabic_ratio(text: str, threshold: float = 0.3) -> bool:
    if not text:
        return False
    ar_count = sum(1 for c in text if "\u0600" <= c <= "\u06FF")
    return ar_count / len(text) >= threshold


class KeyboardConverter(BaseTool):
    def __init__(self):
        super().__init__(
            name="keyboard",
            description=(
                "Convert text between Arabic and English keyboard layouts. "
                "Usage: keyboard: convert hello (if you typed Arabic with English keyboard) "
                "or keyboard: convert هيثتف (if you typed English with Arabic keyboard)"
            ),
        )

    def execute(self, input_data: str = "") -> str:
        cmd = self._clean(input_data)
        if not cmd:
            return self._usage()

        if cmd.startswith("convert "):
            text = cmd[8:].strip()
        elif cmd.startswith("auto "):
            text = cmd[5:].strip()
        else:
            text = cmd

        if not text:
            return self._usage()

        return self.convert(text)

    def convert(self, text: str) -> str:
        if has_arabic_ratio(text, 0.5):
            result = "".join(AR_TO_EN.get(c, c) for c in text)
        else:
            result = "".join(EN_TO_AR.get(c, c) for c in text)

        if result == text:
            return f"No conversion needed: {text}"
        return f"{text} → {result}"

    def _usage(self) -> str:
        return (
            "Usage:\n"
            '  keyboard: convert lklh (English keyboard → Arabic)\n'
            '  keyboard: convert هيثتف (Arabic keyboard → English)\n'
            '  keyboard: auto ... (auto-detect direction)'
        )

    def _clean(self, cmd: str) -> str:
        for prefix in ["keyboard:", "keyboard:"]:
            if cmd.lower().startswith(prefix):
                cmd = cmd[len(prefix):]
        return cmd.strip()
