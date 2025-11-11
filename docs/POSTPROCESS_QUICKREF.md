# 后处理器快速参考

## 快速开始

```python
from ner.postprocess import Postprocessor
from ner.postprocess.rules import BIOConsistencyRule, ConfidenceThresholdRule

# 创建
postprocessor = Postprocessor([
    BIOConsistencyRule(),
    ConfidenceThresholdRule(threshold=0.5)
])

# 使用
tokens, labels, confidences = postprocessor.apply(tokens, labels, confidences)
```

## 内置规则速查

| 规则 | 功能 | 主要参数 |
|------|------|----------|
| `BIOConsistencyRule` | BIO标签一致性 | `fix_orphan_i='to_b'` |
| `ConfidenceThresholdRule` | 置信度过滤 | `threshold=0.5` |
| `EntityBoundaryRule` | 边界修正 | `remove_boundary_punct=True` |
| `MinEntityLengthRule` | 最小长度过滤 | `min_length=2` |
| `PatternCorrectionRule` | 模式匹配修正 | `patterns={}` |

## 从配置加载

```python
from ner.postprocess import build_postprocessor_from_config

config = {
    "postprocess": {
        "rules": [
            {"type": "bio_consistency", "params": {}},
            {"type": "confidence_threshold", "params": {"threshold": 0.5}}
        ]
    }
}

postprocessor = build_postprocessor_from_config(config)
```

## 集成到代码

### 在 evaluator 中
```python
evaluator = NEREvaluator(model, tokenizer, label_list, postprocessor=postprocessor)
```

### 在 predict 中
```python
result = model.predict(text, tokenizer, postprocessor=postprocessor)
```

### 在 REPL 中
```python
self.postprocessor = build_postprocessor_from_file("config.json")
prediction = self.model.predict(text, tokenizer, postprocessor=self.postprocessor)
```

## 自定义规则

```python
from ner.postprocess import BaseRule, PredictionResult

class MyRule(BaseRule):
    name = 'my_rule'
    
    def apply(self, result: PredictionResult) -> PredictionResult:
        # 修改 result.tokens, result.labels, result.confidences
        return result
```

## 推荐组合

**基础**：
```python
[BIOConsistencyRule(), EntityBoundaryRule()]
```

**严格**：
```python
[BIOConsistencyRule(), ConfidenceThresholdRule(0.7), EntityBoundaryRule(), MinEntityLengthRule(2)]
```

**宽松**：
```python
[BIOConsistencyRule()]
```
