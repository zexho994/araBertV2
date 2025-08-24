from __future__ import annotations

from typing import List, Tuple, Optional, Iterable

# 默认的 Unicode 归一化形式
DEFAULT_UNICODE_NORMALIZE_FORM = "NFC"

# 预处理步骤名称
UNICODE_NORMALIZE_STEP = 'unicode_normalize'
WHITESPACE_NORMALIZE_STEP = 'whitespace_normalize'
LOWERCASE_STEP = 'lowercase'
ARABIC_REMOVE_DIACRITICS_STEP = 'arabic_remove_diacritics'
PUNCTUATION_FILTER_STEP = 'punctuation_filter'

class BaseStep:
    """预处理步骤接口

    实现文本或标记级别的方法，根据需要选择。
    """

    name: str = "base"
    is_label_safe: bool = True  # 是否安全处理标记边界/长度

    def apply_text(self, text: str) -> str:
        """应用文本预处理
        
        Args:
            text: 输入文本
            
        Returns:
            str: 处理后的文本
        """
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

    def __str__(self) -> str:
        return f"Preprocessor(steps={', '.join(step.name for step in self.steps)})"

    def add_step(self, step: BaseStep) -> None:
        """添加预处理步骤
        
        Args:
            step: 要添加的预处理步骤
        """
        self.steps.append(step)

    def apply_text(self, text: str) -> str:
        """应用文本预处理
        
        Args:
            text: 输入文本
            
        Returns:
            str: 处理后的文本
        """
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
        """应用标记预处理
        
        Args:
            tokens: 输入标记列表
            labels: 输入标签列表（可选）
            allow_non_label_safe: 是否允许非标签安全步骤
            
        Returns:
            Tuple[List[str], Optional[List[str]]]: 处理后的标记列表和标签列表
        """
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
    """Unicode 归一化步骤
    将文本转换为指定的 Unicode 形式，以确保文本的正确处理和一致性。
    
    Args:
        form: 归一化形式，可选 "NFC" 或 "NFKC"
    """
    name = UNICODE_NORMALIZE_STEP
    is_label_safe = True

    def __init__(self, form: str = DEFAULT_UNICODE_NORMALIZE_FORM) -> None:
        self.form = form

    def apply_text(self, text: str) -> str:
        return unicodedata.normalize(self.form, text) if text else text

    def apply_tokens(self, tokens: List[str], labels: Optional[List[str]] = None):
        if not tokens:
            return tokens, labels
        return [unicodedata.normalize(self.form, t) if t else t for t in tokens], labels


class WhitespaceNormalizeStep(BaseStep):
    """空白归一化步骤
    
    Args:
        collapse: 是否合并连续的空白字符
        trim: 是否去除首尾空白字符
    """
    name = WHITESPACE_NORMALIZE_STEP
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
    """小写化步骤
    
    Args:
        keep_cased: 是否保留大小写（默认 False）
    """
    name = LOWERCASE_STEP
    is_label_safe = True

    def apply_text(self, text: str) -> str:
        return text.lower() if text else text

    def apply_tokens(self, tokens: List[str], labels: Optional[List[str]] = None):
        if not tokens:
            return tokens, labels
        return [t.lower() for t in tokens], labels


class ArabicRemoveDiacriticsStep(BaseStep):
    """阿拉伯语去重音步骤
    
    Args:
        keep_diacritics: 是否保留重音（默认 False）
    """
    name = ARABIC_REMOVE_DIACRITICS_STEP
    is_label_safe = True

    _diacritics = re.compile(r"[\u064B-\u0652\u0670\u0640]")

    def apply_text(self, text: str) -> str:
        return self._diacritics.sub("", text) if text else text

    def apply_tokens(self, tokens: List[str], labels: Optional[List[str]] = None):
        if not tokens:
            return tokens, labels
        return [self._diacritics.sub("", t) for t in tokens], labels


class PunctuationFilterStep(BaseStep):
    """标点过滤步骤
    
    Args:
        keep: 保留的标点列表
        remove: 移除的标点列表
    """
    name = PUNCTUATION_FILTER_STEP
    is_label_safe = True

    def __init__(self, keep: Optional[List[str]] = None, remove: Optional[List[str]] = None) -> None:
        self.keep = set(keep or [])
        self.remove = set(remove or [])

    def _filter(self, s: str) -> str:
        """过滤标点
        
        Args:
            s: 输入字符串
            
        Returns:
            str: 处理后的字符串
        """
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


