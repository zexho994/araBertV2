# NER 后处理模块

## 概述

后处理模块提供规则化的预测结果修正能力，在模型预测后应用规则来提升结果质量。

## 设计理念

- **与预处理器对称**：采用相同的Pipeline设计模式
- **规则可组合**：支持多个规则链式应用
- **配置驱动**：可从配置文件加载规则
- **易于扩展**：继承BaseRule即可添加自定义规则

## 快速开始

### 基本使用

```python
from ner.postprocess import Postprocessor
from ner.postprocess.rules import BIOConsistencyRule, ConfidenceThresholdRule

# 创建后处理器
postprocessor = Postprocessor([
    BIOConsistencyRule(),
    ConfidenceThresholdRule(threshold=0.5)
])

# 应用后处理
tokens = ['John', 'Smith', 'works', 'at', 'Google']
labels = ['B-PER', 'I-PER', 'O', 'O', 'B-ORG']
confidences = [0.9, 0.8, 0.95, 0.92, 0.7]

processed_tokens, processed_labels, processed_confidences = postprocessor.apply(
    tokens, labels, confidences
)
```

### 从配置加载

```python
from ner.postprocess import build_postprocessor_from_config

config = {
    "postprocess": {
        "rules": [
            {"type": "bio_consistency", "params": {"fix_orphan_i": "to_b"}},
            {"type": "confidence_threshold", "params": {"threshold": 0.5}}
        ]
    }
}

postprocessor = build_postprocessor_from_config(config)
```

## 内置规则

详见 rules.py 文件。
