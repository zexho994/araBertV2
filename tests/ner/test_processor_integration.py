import pytest

from src.ner.data.processor import NERDataProcessor
from src.ner.utils.logger import NERLogger


def _minimal_config_with_pipeline():
    """
    最小配置，包含预处理管道
    """
    return {
        "data": {
            "max_length": 16,
            "preprocessing_pipeline": [
                {"step": "unicode_normalize", "params": {"form": "NFC"}},
                {"step": "whitespace_normalize", "params": {"collapse": True, "trim": True}},
                {"step": "arabic_diacritics_filter"},
            ],
            "encoding": "utf-8",
        },
        "labels": {
            # 使用 label_mapping 直接定义映射（与项目配置保持一致）
            "label_mapping": {
                "O": 0,
                "B-STREET": 1,
                "I-STREET": 2,
                "B-CITY": 3,
                "I-CITY": 4,
            }
        }
    }


def test_processor_preprocess_example_label_safe():
    """
    测试处理器预处理示例
    """
    logger = NERLogger(log_to_file=False, log_to_console=False)
    config = _minimal_config_with_pipeline()
    processor = NERDataProcessor(config, logger)

    ex = {"tokens": ["  شارع", "الملك   ", "   فهد  "], "labels": ["B-STREET", "I-STREET", "I-STREET"]}
    out = processor._preprocess_example(ex)
    assert out is not None
    assert out["tokens"] == ["شارع", "الملك", "فهد"]
    assert out["labels"] == ["B-STREET", "I-STREET", "I-STREET"]
    assert out["text"] == "شارع الملك فهد"


def test_processor_truncate_respects_max_length():
    """
    测试处理器截断示例，max_length 应该被严格遵守
    """
    logger = NERLogger(log_to_file=False, log_to_console=False)
    config = _minimal_config_with_pipeline()
    config["data"]["max_length"] = 2
    processor = NERDataProcessor(config, logger)

    ex = {"tokens": ["A", "B", "C"], "labels": ["O", "O", "B-CITY"]}
    out = processor._preprocess_example(ex)
    assert out is not None
    assert len(out["tokens"]) == 2
    assert len(out["labels"]) == 2


