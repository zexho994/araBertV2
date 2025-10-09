from __future__ import annotations

from typing import Dict, Any, List

from .pipeline import (
    Preprocessor,
    BaseStep,
    UnicodeNormalizeStep,
    WhitespaceNormalizeStep,
    LowercaseStep,
    ArabicDiacriticsFilterStep,
    PunctuationFilterStep,
    EmojiFilterStep,
    DigitNormalizeStep,
    PunctuationNormalizeStep,
    SpecialPunctuationSpacingStep,
    UNICODE_NORMALIZE_STEP,
    WHITESPACE_NORMALIZE_STEP,
    LOWERCASE_STEP,
    ARABIC_DIACRITICS_FILTER_STEP,
    PUNCTUATION_FILTER_STEP,
    EMOJI_FILTER_STEP,
    DIGIT_NORMALIZE_STEP,
    PUNCTUATION_NORMALIZE_STEP,
    SPECIAL_PUNCT_SPACING_STEP,
)


def _instantiate_step(spec: Dict[str, Any]) -> BaseStep:
    """
    实例化预处理步骤
    
    Args:
        spec: 步骤配置字典
        
    Returns:
        BaseStep: 实例化的步骤对象
    """
    name = spec.get("step") or spec.get("name")
    params = spec.get("params", {})


    if name == UNICODE_NORMALIZE_STEP:
        return UnicodeNormalizeStep(**params)
    if name == WHITESPACE_NORMALIZE_STEP:
        return WhitespaceNormalizeStep(**params)
    if name == LOWERCASE_STEP:
        return LowercaseStep(**params)
    if name == ARABIC_DIACRITICS_FILTER_STEP:
        return ArabicDiacriticsFilterStep(**params)
    if name == PUNCTUATION_FILTER_STEP:
        return PunctuationFilterStep(**params)
    if name == EMOJI_FILTER_STEP:
        return EmojiFilterStep(**params)
    if name == DIGIT_NORMALIZE_STEP:
        return DigitNormalizeStep(**params)
    if name == PUNCTUATION_NORMALIZE_STEP:
        return PunctuationNormalizeStep(**params)
    if name == SPECIAL_PUNCT_SPACING_STEP:
        return SpecialPunctuationSpacingStep(**params)

    raise ValueError(f"Unknown preprocessing step: {name}")


def build_preprocessor_from_config(config: Dict[str, Any]) -> Preprocessor:
    """ 从配置构建 Preprocessor，支持向后兼容

    优先级：
    1) data.preprocessing_pipeline (步骤列表)
    2) data.preprocessing (布尔开关)
    """
    data_cfg = (config or {}).get("data", {})
    pipeline_spec = data_cfg.get("preprocessing_pipeline")
    if isinstance(pipeline_spec, list) and pipeline_spec:
        steps: List[BaseStep] = [_instantiate_step(item) for item in pipeline_spec]
        return Preprocessor(steps)
    else:
        raise ValueError("No preprocessing pipeline found in config")


