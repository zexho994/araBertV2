# NER 后处理器设计文档

## 1. 设计目标

创建一个通用、可扩展的后处理模块，用于在NER模型预测后应用规则修正，提升预测质量。

## 2. 核心设计原则

### 2.1 与预处理器对称
- 采用相同的Pipeline + Step设计模式
- 保持API一致性，降低学习成本

### 2.2 操作对象
- 输入：`(tokens, labels, confidences)` 三元组
- 输出：处理后的 `(tokens, labels, confidences)` 三元组
- 使用 `PredictionResult` 数据类封装，确保数据一致性

### 2.3 规则可组合
- 规则按顺序链式应用
- 每个规则独立、可测试
- 支持动态添加规则

### 2.4 配置驱动
- 支持从JSON配置加载
- 规则参数可配置
- 便于不同场景使用不同规则组合

## 3. 模块结构

```
src/ner/postprocess/
├── __init__.py           # 公共API
├── pipeline.py           # 核心类：Postprocessor, BaseRule, PredictionResult
├── rules.py              # 内置规则实现
├── builder.py            # 配置构建器
└── README.md             # 使用文档
```

## 4. 核心类设计

### 4.1 PredictionResult
```python
@dataclass
class PredictionResult:
    tokens: List[str]
    labels: List[str]
    confidences: List[float]
```
- 封装预测结果
- 自动验证数据一致性
- 便于规则间传递

### 4.2 BaseRule
```python
class BaseRule:
    name: str
    description: str
    
    def apply(self, result: PredictionResult) -> PredictionResult:
        pass
```
- 所有规则的基类
- 子类只需实现 `apply` 方法
- 支持任意复杂的修正逻辑

### 4.3 Postprocessor
```python
class Postprocessor:
    def __init__(self, rules: List[BaseRule])
    def apply(self, tokens, labels, confidences) -> Tuple
    def apply_batch(self, batch_tokens, batch_labels, batch_confidences) -> Tuple
```
- 管理规则链
- 提供单样本和批量处理接口
- 按顺序应用所有规则

## 5. 内置规则

### 5.1 BIOConsistencyRule
- **功能**：确保BIO标签一致性
- **修正**：孤立的I标签、类型不匹配
- **参数**：`fix_orphan_i` (to_b/to_o)

### 5.2 ConfidenceThresholdRule
- **功能**：过滤低置信度预测
- **修正**：将低于阈值的标签设为O
- **参数**：`threshold`, `keep_entity_if_any_high`

### 5.3 EntityBoundaryRule
- **功能**：修正实体边界
- **修正**：移除边界的标点、停用词
- **参数**：`remove_boundary_punct`, `remove_boundary_stopwords`

### 5.4 MinEntityLengthRule
- **功能**：过滤过短实体
- **修正**：移除长度小于阈值的实体
- **参数**：`min_length`, `min_length_by_type`

### 5.5 PatternCorrectionRule
- **功能**：基于正则模式修正
- **修正**：根据模式修正实体类型
- **参数**：`patterns` (dict)

### 5.6 MergeAdjacentRule
- **功能**：合并相邻同类实体
- **修正**：跨越连接符合并实体
- **参数**：`merge_across_tokens`, `max_gap`

## 6. 集成方式

### 6.1 在 evaluator.py 中
```python
# 初始化时传入
evaluator = NEREvaluator(
    model, tokenizer, label_list,
    postprocessor=postprocessor
)

# 评估时应用
if self.postprocessor:
    y_pred_sequences, _, _ = self.postprocessor.apply_batch(...)
```

### 6.2 在 model.predict 中
```python
def predict_tokens(self, ..., postprocessor=None):
    # 预测
    # ...
    
    # 后处理
    if postprocessor:
        words, labels, confidences = postprocessor.apply(...)
    
    return result
```

### 6.3 在 REPL 中
```python
# 加载时从配置读取
self.postprocessor = build_postprocessor_from_file(config_path)

# 预测时传入
prediction = self.model.predict(..., postprocessor=self.postprocessor)
```

## 7. 配置示例

```json
{
  "postprocess": {
    "rules": [
      {
        "type": "bio_consistency",
        "params": {"fix_orphan_i": "to_b"}
      },
      {
        "type": "confidence_threshold",
        "params": {"threshold": 0.5}
      },
      {
        "type": "entity_boundary",
        "params": {
          "remove_boundary_punct": true
        }
      }
    ]
  }
}
```

## 8. 扩展性

### 8.1 添加自定义规则
```python
class MyCustomRule(BaseRule):
    name = 'my_rule'
    
    def apply(self, result: PredictionResult) -> PredictionResult:
        # 实现逻辑
        return result

# 注册
RULE_FACTORY['my_rule'] = MyCustomRule
```

### 8.2 规则组合策略
- 可以根据不同场景组合不同规则
- 训练时：不使用后处理
- 评估时：可选使用
- 预测时：默认使用

## 9. 优势总结

1. **通用性**：统一接口，适用于所有预测场景
2. **灵活性**：规则可自由组合，按需启用
3. **可维护性**：规则独立，易于测试和调试
4. **可配置性**：配置文件管理，无需修改代码
5. **性能友好**：支持批量处理，不影响效率
6. **易扩展**：添加新规则只需继承BaseRule
7. **类型安全**：使用dataclass确保数据一致性

## 10. 使用建议

### 10.1 推荐规则组合

**基础组合**（适用于大多数场景）：
```python
Postprocessor([
    BIOConsistencyRule(),
    EntityBoundaryRule(remove_boundary_punct=True)
])
```

**严格组合**（高精度场景）：
```python
Postprocessor([
    BIOConsistencyRule(),
    ConfidenceThresholdRule(threshold=0.7),
    EntityBoundaryRule(remove_boundary_punct=True),
    MinEntityLengthRule(min_length=2)
])
```

**宽松组合**（高召回场景）：
```python
Postprocessor([
    BIOConsistencyRule(),
    MergeAdjacentRule()
])
```

### 10.2 性能考虑
- 规则按顺序执行，将快速规则放前面
- 批量处理时使用 `apply_batch`
- 避免在规则中进行重复计算

### 10.3 调试建议
- 单独测试每个规则
- 使用日志记录规则应用前后的变化
- 在开发环境启用详细输出
