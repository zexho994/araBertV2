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
)


def _instantiate_step(spec: Dict[str, Any]) -> BaseStep:
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

    raise ValueError(f"Unknown preprocessing step: {name}")


def build_preprocessor_from_config(config: Dict[str, Any]) -> Preprocessor:
    """Build a Preprocessor from config with backward compatibility.

    Priority:
    1) data.preprocessing_pipeline (list of steps)
    2) data.preprocessing (legacy booleans)
    """
    data_cfg = (config or {}).get("data", {})

    pipeline_spec = data_cfg.get("preprocessing_pipeline")
    if isinstance(pipeline_spec, list) and pipeline_spec:
        steps: List[BaseStep] = [_instantiate_step(item) for item in pipeline_spec]
        return Preprocessor(steps)

    # Fallback to legacy booleans
    legacy = data_cfg.get("preprocessing", {}) or {}
    steps: List[BaseStep] = []

    # order: unicode -> arabic diacritics -> whitespace -> lowercase -> punctuation
    # clean_text implies unicode + whitespace trim/collapse
    if legacy.get("clean_text", False):
        steps.append(UnicodeNormalizeStep())
        steps.append(WhitespaceNormalizeStep(collapse=True, trim=True))

    if legacy.get("normalize_arabic", False):
        # placeholder: use diacritics removal as safe normalization baseline
        steps.append(ArabicRemoveDiacriticsStep())

    if legacy.get("remove_diacritics", False):
        steps.append(ArabicRemoveDiacriticsStep())

    # Mixed script normalize is not implemented as a separate step yet.

    # Lowercase only if explicitly requested in legacy (not common for Arabic)
    if legacy.get("lowercase", False):
        steps.append(LowercaseStep())

    if legacy.get("remove_special_chars", False):
        steps.append(PunctuationFilterStep())

    return Preprocessor(steps)


