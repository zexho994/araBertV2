from __future__ import annotations

from typing import List, Tuple, Optional, Dict, Any, Iterable


class BaseStep:
    """Preprocessing step interface.

    Implement either text or token-level methods as needed.
    """

    name: str = "base"
    is_label_safe: bool = True  # does not change token boundaries/lengths

    def apply_text(self, text: str) -> str:
        return text

    def apply_tokens(
        self,
        tokens: List[str],
        labels: Optional[List[str]] = None,
    ) -> Tuple[List[str], Optional[List[str]]]:
        return tokens, labels


class Preprocessor:
    """Composable pipeline of preprocessing steps.

    - text mode: apply on raw text
    - tokens mode: apply on (tokens, labels), optionally enforcing label-safe only
    """

    def __init__(self, steps: Optional[Iterable[BaseStep]] = None):
        self.steps: List[BaseStep] = list(steps) if steps else []

    def add_step(self, step: BaseStep) -> None:
        self.steps.append(step)

    def apply_text(self, text: str) -> str:
        processed = text
        for step in self.steps:
            processed = step.apply_text(processed)
        return processed

    def apply_tokens(
        self,
        tokens: List[str],
        labels: Optional[List[str]] = None,
        allow_non_label_safe: bool = False,
    ) -> Tuple[List[str], Optional[List[str]]]:
        processed_tokens = tokens
        processed_labels = labels
        for step in self.steps:
            if step.is_label_safe or allow_non_label_safe:
                processed_tokens, processed_labels = step.apply_tokens(processed_tokens, processed_labels)
        return processed_tokens, processed_labels


# Built-in steps (label-safe) -------------------------------------------------

import re
import unicodedata


class UnicodeNormalizeStep(BaseStep):
    name = "unicode_normalize"
    is_label_safe = True

    def __init__(self, form: str = "NFC") -> None:
        self.form = form

    def apply_text(self, text: str) -> str:
        return unicodedata.normalize(self.form, text) if text else text

    def apply_tokens(self, tokens: List[str], labels: Optional[List[str]] = None):
        if not tokens:
            return tokens, labels
        return [unicodedata.normalize(self.form, t) if t else t for t in tokens], labels


class WhitespaceNormalizeStep(BaseStep):
    name = "whitespace_normalize"
    is_label_safe = True

    def __init__(self, collapse: bool = True, trim: bool = True) -> None:
        self.collapse = collapse
        self.trim = trim

    def _normalize(self, s: str) -> str:
        out = s
        if self.collapse:
            out = re.sub(r"\s+", " ", out)
        if self.trim:
            out = out.strip()
        return out

    def apply_text(self, text: str) -> str:
        return self._normalize(text) if text else text

    def apply_tokens(self, tokens: List[str], labels: Optional[List[str]] = None):
        if not tokens:
            return tokens, labels
        return [self._normalize(t) for t in tokens], labels


class LowercaseStep(BaseStep):
    name = "lowercase"
    is_label_safe = True

    def apply_text(self, text: str) -> str:
        return text.lower() if text else text

    def apply_tokens(self, tokens: List[str], labels: Optional[List[str]] = None):
        if not tokens:
            return tokens, labels
        return [t.lower() for t in tokens], labels


class ArabicRemoveDiacriticsStep(BaseStep):
    name = "arabic_remove_diacritics"
    is_label_safe = True

    _diacritics = re.compile(r"[\u064B-\u0652\u0670\u0640]")

    def apply_text(self, text: str) -> str:
        return self._diacritics.sub("", text) if text else text

    def apply_tokens(self, tokens: List[str], labels: Optional[List[str]] = None):
        if not tokens:
            return tokens, labels
        return [self._diacritics.sub("", t) for t in tokens], labels


class PunctuationFilterStep(BaseStep):
    name = "punctuation_filter"
    is_label_safe = True

    def __init__(self, keep: Optional[List[str]] = None, remove: Optional[List[str]] = None) -> None:
        self.keep = set(keep or [])
        self.remove = set(remove or [])

    def _filter(self, s: str) -> str:
        # Keep tokens intact: remove only characters, not split/merge
        out_chars = []
        for ch in s:
            cat = unicodedata.category(ch)
            if ch in self.keep:
                out_chars.append(ch)
            elif self.remove and ch in self.remove:
                continue
            elif cat.startswith("P"):
                continue
            else:
                out_chars.append(ch)
        return "".join(out_chars)

    def apply_text(self, text: str) -> str:
        return self._filter(text) if text else text

    def apply_tokens(self, tokens: List[str], labels: Optional[List[str]] = None):
        if not tokens:
            return tokens, labels
        return [self._filter(t) for t in tokens], labels


