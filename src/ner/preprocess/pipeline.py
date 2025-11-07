from __future__ import annotations

from typing import List, Tuple, Optional, Iterable

# 默认的 Unicode 归一化形式
DEFAULT_UNICODE_NORMALIZE_FORM = "NFC"

# 预处理步骤名称
## 归一
LOWERCASE_STEP = 'lowercase'
UNICODE_NORMALIZE_STEP = 'unicode_normalize'
WHITESPACE_NORMALIZE_STEP = 'whitespace_normalize'
DIGIT_NORMALIZE_STEP = 'digit_normalize'
PUNCTUATION_NORMALIZE_STEP = 'punctuation_normalize'

## 去噪
ARABIC_DIACRITICS_FILTER_STEP = 'arabic_diacritics_filter'
PUNCTUATION_FILTER_STEP = 'punctuation_filter'
EMOJI_FILTER_STEP = 'emoji_filter'

## 结构优化
SPECIAL_PUNCT_SPACING_STEP = 'special_punct_spacing'

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
    """Unicode 归一化
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
    """去噪：合并连续的空白字符，去除首尾空白字符
    
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
    """小写归一化
    
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


class ArabicDiacriticsFilterStep(BaseStep):
    """去噪: 阿拉伯语去重音
    
    Args:
        keep_diacritics: 是否保留重音（默认 False）
    """
    name = ARABIC_DIACRITICS_FILTER_STEP
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

    # 默认保留的标点类别
    # 这些类别通常是可见字符，如括号、引号、连字符等
    DEFAULT_KEEP_PUNCTUATION_CATEGORIES = ['.','(',')','[',']','|','\\','\'']

    # 默认移除的标点类别
    # 这些类别通常是控制字符或不可见字符，如换行符、制表符等
    DEFAULT_REMOVE_PUNCTUATION_CATEGORIES = ['~', '?', '!', ':', '-', ',', '，', '/']
    
    # 默认需要去重的重复标点符号
    # 这些标点符号如果连续出现会被合并为单个字符
    DEFAULT_REMOVE_REPEAT_PUNCTUATION = [',','-','.']
    
    # 默认在两边有空格时需要移除的标点符号
    # 这些标点符号如果前后都有空格就会被移除
    DEFAULT_REMOVE_IF_SPACED = ['-']

    def __init__(self, keep: Optional[List[str]] = None, remove: Optional[List[str]] = None, 
                 remove_repeat: Optional[List[str]] = None, remove_if_spaced: Optional[List[str]] = None) -> None:
        self.keep = set(keep or self.DEFAULT_KEEP_PUNCTUATION_CATEGORIES)
        self.remove = set(remove or self.DEFAULT_REMOVE_PUNCTUATION_CATEGORIES)
        self.remove_repeat = set(remove_repeat or self.DEFAULT_REMOVE_REPEAT_PUNCTUATION)
        self.remove_if_spaced = set(remove_if_spaced or self.DEFAULT_REMOVE_IF_SPACED)

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
                out_chars.append(" ")  # 用空格替换移除的标点
            elif cat.startswith("P"):
                out_chars.append(" ")  # 用空格替换其他标点
            else:
                out_chars.append(ch)
        
        result = "".join(out_chars)
        
        # 处理重复标点符号
        for punct in self.remove_repeat:
            # 使用正则表达式将连续的重复标点替换为单个标点
            pattern = re.escape(punct) + r"{2,}"
            result = re.sub(pattern, punct, result)
        
        # 处理两边有空格的标点符号移除
        for punct in self.remove_if_spaced:
            # 使用正则表达式匹配前后都有空格的标点符号并移除
            pattern = r"\s+" + re.escape(punct) + r"\s+"
            result = re.sub(pattern, " ", result)
        
        return result

    def apply_text(self, text: str) -> str:
        return self._filter(text) if text else text

    def apply_tokens(self, tokens: List[str], labels: Optional[List[str]] = None):
        if not tokens:
            return tokens, labels
        # 在 token 级别，用空格替换所有标点以保持标签对齐的稳定性
        def _strip_punct_token(s: str) -> str:
            out_chars = []
            for ch in s:
                # 无论 keep/remove 设置，token 级别一律用空格替换所有 Unicode 标点类别
                if unicodedata.category(ch).startswith("P"):
                    out_chars.append(" ")  # 用空格替换标点
                else:
                    out_chars.append(ch)
            result = "".join(out_chars)
            
            # 处理重复标点符号（在token级别也应用去重逻辑）
            for punct in self.remove_repeat:
                pattern = re.escape(punct) + r"{2,}"
                result = re.sub(pattern, punct, result)
            
            # 处理两边有空格的标点符号移除（在token级别也应用相同逻辑）
            for punct in self.remove_if_spaced:
                pattern = r"\s+" + re.escape(punct) + r"\s+"
                result = re.sub(pattern, " ", result)
            
            return result

        return [_strip_punct_token(t) for t in tokens], labels


class EmojiFilterStep(BaseStep):
    """去噪: 过滤 emoji 表情符号
    
    移除文本中的 emoji 表情，包括各种 Unicode emoji 范围：
    - 情绪图标、符号和象形文字
    - 交通和地图符号
    - 补充符号和象形文字
    - 杂项符号等
    """
    name = EMOJI_FILTER_STEP
    is_label_safe = True

    # Emoji Unicode 范围正则表达式
    _emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # 情绪图标 (Emoticons) 😀😁😂
        "\U0001F300-\U0001F5FF"  # 符号和象形文字 (Symbols & Pictographs) 🌍
        "\U0001F680-\U0001F6FF"  # 交通和地图符号 (Transport & Map Symbols) 🚗🚕🚙
        "\U0001F700-\U0001F77F"  # 炼金术符号 (Alchemical Symbols) 🔮
        "\U0001F780-\U0001F7FF"  # 几何形状扩展 (Geometric Shapes Extended)
        "\U0001F800-\U0001F8FF"  # 补充箭头-C (Supplemental Arrows-C) ➡️➡️➡️
        "\U0001F900-\U0001F9FF"  # 补充符号和象形文字 (Supplemental Symbols and Pictographs) 🤝🤝🤝
        "\U0001FA00-\U0001FA6F"  # 象棋符号 (Chess Symbols) 🎯
        "\U0001FA70-\U0001FAFF"  # 符号和象形文字扩展-A (Symbols and Pictographs Extended-A)
        "\U00002600-\U000026FF"  # 杂项符号 (Miscellaneous Symbols)
        "\U00002700-\U000027BF"  # 装饰符号 (Dingbats)
        "\U0001F1E0-\U0001F1FF"  # 旗帜 (Flags)
        "\U00002300-\U000023FF"  # 杂项技术符号 (Miscellaneous Technical)
        "\U0001F000-\U0001F02F"  # 麻将牌 (Mahjong Tiles)
        "\U0001F0A0-\U0001F0FF"  # 扑克牌 (Playing Cards)
        "\U0001F100-\U0001F1FF"  # 封闭字母数字补充 (Enclosed Alphanumeric Supplement)
        "]+",
        flags=re.UNICODE
    )

    def apply_text(self, text: str) -> str:
        return self._emoji_pattern.sub("", text) if text else text

    def apply_tokens(self, tokens: List[str], labels: Optional[List[str]] = None):
        if not tokens:
            return tokens, labels
        return [self._emoji_pattern.sub("", t) for t in tokens], labels



class DigitNormalizeStep(BaseStep):
    """数字归一化：将阿拉伯-印地数字统一转换为 0-9
    
    - 支持范围：U+0660–U+0669, U+06F0–U+06F9
    """
    name = DIGIT_NORMALIZE_STEP
    is_label_safe = True

    _trans = str.maketrans({
        # Arabic-Indic digits
        "\u0660": "0", "\u0661": "1", "\u0662": "2", "\u0663": "3", "\u0664": "4",
        "\u0665": "5", "\u0666": "6", "\u0667": "7", "\u0668": "8", "\u0669": "9",
        # Eastern Arabic-Indic digits
        "\u06F0": "0", "\u06F1": "1", "\u06F2": "2", "\u06F3": "3", "\u06F4": "4",
        "\u06F5": "5", "\u06F6": "6", "\u06F7": "7", "\u06F8": "8", "\u06F9": "9",
    })

    def apply_text(self, text: str) -> str:
        return text.translate(self._trans) if text else text

    def apply_tokens(self, tokens: List[str], labels: Optional[List[str]] = None):
        if not tokens:
            return tokens, labels
        return [t.translate(self._trans) for t in tokens], labels


class PunctuationNormalizeStep(BaseStep):
    """标点统一：规范化常见变体

    示例：
      - '،' → ','
      - 多个连字符 '--'、'—'、'–'、'−' → '-'
      - 省略号 '…' → '...'
    """
    name = PUNCTUATION_NORMALIZE_STEP
    is_label_safe = True

    _char_map = str.maketrans({
        # Arabic punctuation to Latin
        "\u060C": ",",   # Arabic comma → ,
        "\u061B": ";",   # Arabic semicolon → ;
        "\u061F": "?",   # Arabic question mark → ?
        # Dashes and hyphens to simple hyphen-minus
        "\u2010": "-",  # hyphen
        "\u2011": "-",  # non-breaking hyphen
        "\u2012": "-",  # figure dash
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
        "\u2015": "-",  # horizontal bar
        "\u2212": "-",  # minus sign
        # Quotes (optional normalization)
        # "\u2018": "'", "\u2019": "'", "\u201A": "'",
        # "\u201C": '"', "\u201D": '"', "\u201E": '"',
        # Ellipsis
        # "\u2026": "...",
    })

    _multi_hyphens = re.compile(r"-{2,}")

    def _unify(self, s: str) -> str:
        if not s:
            return s
        out = s.translate(self._char_map)
        out = self._multi_hyphens.sub("-", out)
        return out

    def apply_text(self, text: str) -> str:
        return self._unify(text) if text else text

    def apply_tokens(self, tokens: List[str], labels: Optional[List[str]] = None):
        if not tokens:
            return tokens, labels
        return [self._unify(t) for t in tokens], labels


class SpecialPunctuationSpacingStep(BaseStep):
    """特殊标点符号处理：在关键分隔符前后添加空格

    默认处理的分隔符：`,`
    """
    name = SPECIAL_PUNCT_SPACING_STEP
    is_label_safe = True

    # match separators with optional surrounding spaces
    _sep_pattern = re.compile(r"\s*([,])\s*")
    _spaces_collapse = re.compile(r"\s{2,}")

    def _space_around(self, s: str) -> str:
        if not s:
            return s
        out = self._sep_pattern.sub(r" \1 ", s)
        out = self._spaces_collapse.sub(" ", out)
        return out.strip()

    def apply_text(self, text: str) -> str:
        return self._space_around(text) if text else text

    def apply_tokens(self, tokens: List[str], labels: Optional[List[str]] = None):
        if not tokens:
            return tokens, labels
        return [self._space_around(t) for t in tokens], labels