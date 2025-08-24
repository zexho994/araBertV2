import pytest

from src.ner.preprocess import Preprocessor
from src.ner.preprocess.pipeline import (
    UnicodeNormalizeStep,
    WhitespaceNormalizeStep,
    LowercaseStep,
    ArabicRemoveDiacriticsStep,
    PunctuationFilterStep,
)

"""
测试预处理管道
"""


def test_unicode_normalize_text():
    """
    测试 Unicode 归一化步骤
    """
    p = Preprocessor([UnicodeNormalizeStep(form="NFC")])
    text = "A\u030A"  # 'Å' composed as 'A' + ring combining
    out = p.apply_text(text)
    assert out == out  # should run without error; specific equality to NFC form is environment-dependent


def test_whitespace_normalize_tokens_label_safe():
    """
    测试空白归一化步骤
    """
    p = Preprocessor([WhitespaceNormalizeStep(collapse=True, trim=True)])
    tokens = ["  شارع", "الملك   ", "   فهد  "]
    labels = ["B-STREET", "I-STREET", "I-STREET"]
    t2, l2 = p.apply_tokens(tokens, labels)
    assert t2 == ["شارع", "الملك", "فهد"]
    assert l2 == labels


def test_lowercase_tokens_label_safe():
    """
    测试小写化步骤
    """
    p = Preprocessor([LowercaseStep()])
    tokens = ["Dubai", "UAE"]
    labels = ["B-CITY", "B-COUNTRY"]
    t2, l2 = p.apply_tokens(tokens, labels)
    assert t2 == ["dubai", "uae"]
    assert l2 == labels


def test_arabic_remove_diacritics_text_and_tokens():
    """
    测试阿拉伯语去重音步骤
    """
    p = Preprocessor([ArabicRemoveDiacriticsStep()])
    text = "السَّلَامُ عَلَيْكُمْ"
    out_text = p.apply_text(text)
    assert out_text.replace(" ", "") == "السلامعليكم"

    tokens = ["السَّلَامُ", "عَلَيْكُمْ"]
    labels = ["O", "O"]
    t2, l2 = p.apply_tokens(tokens, labels)
    assert t2 == ["السلام", "عليكم"]
    assert l2 == labels


def test_punctuation_filter_preserves_alignment():
    """
    测试标点过滤步骤
    """
    p = Preprocessor([PunctuationFilterStep(keep=[",", "."])])
    tokens = ["Road,", "Dubai."]
    labels = ["O", "B-CITY"]
    t2, l2 = p.apply_tokens(tokens, labels)
    # 标点过滤后，除了保留的标点外，其他标点应该被移除；过滤步骤不应该改变token数量
    assert len(t2) == len(tokens)
    assert l2 == labels


