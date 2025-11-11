# 后处理器集成指南

## 设计方案总结

### 1. 模块结构
```
src/ner/postprocess/
├── __init__.py           # 公共API
├── pipeline.py           # 核心Pipeline和BaseRule
├── rules.py              # 内置规则实现
├── builder.py            # 从配置构建
└── README.md             # 使用文档
```

### 2. 核心特性

- **统一接口**：`apply(tokens, labels, confidences)` 返回处理后的三元组
- **批量支持**：`apply_batch()` 用于evaluator的批量处理
- **规则可组合**：多个规则按顺序应用
- **配置驱动**：支持从JSON配置加载

### 3. 集成方式

#### 在 evaluator.py 中集成

```python
# 在 NEREvaluator.__init__ 中添加
def __init__(
    self, 
    model, 
    tokenizer, 
    label_list: List[str],
    device: Optional[torch.device] = None,
    logger: Optional[NERLogger] = None,
    postprocessor: Optional[Postprocessor] = None  # 新增
):
    # ... 现有代码 ...
    self.postprocessor = postprocessor

# 在 evaluate_dataloader 中应用
def evaluate_dataloader(self, dataloader, confidence_threshold: Optional[float] = None):
    # ... 现有预测代码 ...
    
    # 在收集完 y_pred_sequences 后，应用后处理
    if self.postprocessor:
        y_pred_sequences, _, _ = self.postprocessor.apply_batch(
            y_pred_sequences,
            y_pred_sequences,  # labels
            [[1.0] * len(seq) for seq in y_pred_sequences]  # dummy confidences
        )
    
    # ... 继续计算指标 ...
```

#### 在 model.py 的 predict 方法中集成

```python
# 在 NERModel.predict_tokens 中添加参数
def predict_tokens(
    self,
    words: List[str],
    tokenizer=None,
    confidence_threshold: float = 0.5,
    device: Optional[torch.device] = None,
    postprocessor: Optional[Postprocessor] = None  # 新增
) -> Dict[str, Any]:
    # ... 现有预测代码 ...
    
    # 在得到 word_predictions 和 word_confidences 后
    if postprocessor:
        words, word_predictions, word_confidences = postprocessor.apply(
            words, word_predictions, word_confidences
        )
    
    # 然后提取实体
    entities = self._extract_entities(words, word_predictions, word_confidences)
    # ...
```

#### 在 repl_predict.py 中集成

```python
class PredictREPL:
    def __init__(self, logger=None) -> None:
        # ... 现有代码 ...
        self.postprocessor = None  # 新增
    
    def _handle_load(self, model_path: str) -> None:
        # ... 加载模型代码 ...
        
        # 尝试从模型目录加载后处理配置
        config_path = Path(model_path) / "postprocess_config.json"
        if config_path.exists():
            from ner.postprocess import build_postprocessor_from_file
            self.postprocessor = build_postprocessor_from_file(str(config_path))
            self._log_info("已加载后处理配置")
    
    def _handle_predict(self, text: str) -> None:
        # ... 现有代码 ...
        prediction = self.model.predict(
            input_text,
            tokenizer=self.tokenizer,
            confidence_threshold=self.confidence_threshold,
            postprocessor=self.postprocessor  # 传入后处理器
        )
        # ...
```

### 4. 配置示例

在国家配置文件中添加后处理规则（如 `data/ner/configs/countries/uae.json`）：

```json
{
  "country": "uae",
  "preprocess": { ... },
  "postprocess": {
    "rules": [
      {
        "type": "bio_consistency",
        "params": {
          "fix_orphan_i": "to_b"
        }
      },
      {
        "type": "entity_boundary",
        "params": {
          "remove_boundary_punct": true,
          "remove_boundary_stopwords": false
        }
      },
      {
        "type": "min_entity_length",
        "params": {
          "min_length": 2,
          "min_length_by_type": {
            "PER": 2,
            "ORG": 2
          }
        }
      }
    ]
  }
}
```

### 5. 使用流程

1. **训练时**：不使用后处理（保持原始标签）
2. **评估时**：可选使用后处理（通过参数控制）
3. **预测时**：默认使用后处理（提升用户体验）

### 6. 扩展性

添加自定义规则只需：

```python
from ner.postprocess import BaseRule, PredictionResult

class MyCustomRule(BaseRule):
    name = 'my_custom_rule'
    description = 'My custom post-processing rule'
    
    def apply(self, result: PredictionResult) -> PredictionResult:
        # 实现你的逻辑
        return result

# 注册到工厂
from ner.postprocess.builder import RULE_FACTORY
RULE_FACTORY['my_custom_rule'] = MyCustomRule
```

## 优势

1. **通用性**：统一的接口可在任何预测场景使用
2. **灵活性**：规则可自由组合，按需启用
3. **可维护性**：规则独立，易于测试和调试
4. **可配置性**：通过配置文件管理，无需修改代码
5. **性能友好**：支持批量处理，不影响评估效率
