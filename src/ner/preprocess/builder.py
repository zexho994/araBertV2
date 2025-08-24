from __future__ import annotations

from typing import Dict, Any, List

from .pipeline import (
    Preprocessor,
    BaseStep,
    UnicodeNormalizeStep,
    WhitespaceNormalizeStep,
    LowercaseStep,
    ArabicRemoveDiacriticsStep,
    PunctuationFilterStep,
    DigitNormalizeStep,
    PunctuationUnifyStep,
    SpecialPunctuationSpacingStep,
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

    if name == "unicode_normalize":
        return UnicodeNormalizeStep(**params)
    if name == "whitespace_normalize":
        return WhitespaceNormalizeStep(**params)
    if name == "lowercase":
        return LowercaseStep(**params)
    if name == "arabic_remove_diacritics":
        return ArabicRemoveDiacriticsStep(**params)
    if name == "punctuation_filter":
        return PunctuationFilterStep(**params)
    if name == "digit_normalize":
        return DigitNormalizeStep(**params)
    if name == "punctuation_unify":
        return PunctuationUnifyStep(**params)
    if name == "special_punct_spacing":
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
        # 兜底：布尔开关
        legacy = data_cfg.get("preprocessing", {}) or {}
        steps: List[BaseStep] = []

        if legacy.get("clean_text", False):
            # 使用去重音作为安全归一化的基础
            steps.append(UnicodeNormalizeStep())
            steps.append(WhitespaceNormalizeStep(collapse=True, trim=True))

        if legacy.get("normalize_arabic", False):
            # 使用去重音作为安全归一化的基础
            steps.append(ArabicRemoveDiacriticsStep())

        if legacy.get("remove_diacritics", False):
            # 使用去重音作为安全归一化的基础
            steps.append(ArabicRemoveDiacriticsStep())

        # 混合文字系统归一化未实现为单独的步骤

        # 仅在 legacy 中显式请求小写化（阿拉伯语中不常见）
        if legacy.get("lowercase", False):
            steps.append(LowercaseStep())

        if legacy.get("remove_special_chars", False):
            steps.append(PunctuationFilterStep())

        return Preprocessor(steps)


