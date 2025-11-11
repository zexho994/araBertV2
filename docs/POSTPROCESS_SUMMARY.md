# 后处理器配置总结

## 📍 配置位置

后处理规则配置在**国家配置文件**中：

```
data/ner/configs/countries/{country}.json
```

例如：`data/ner/configs/countries/uae.json`

## 🎯 快速配置

### 最小配置（推荐所有场景）

在国家配置文件中添加：

```json
{
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
          "remove_boundary_punct": true
        }
      }
    ]
  }
}
```

### UAE地址识别配置（已配置）

当前 `uae.json` 已配置：

```json
{
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
            "BUILDING": 2,
            "STREET": 2
          }
        }
      }
    ]
  }
}
```

## 🔧 配置加载方式

### 自动加载（推荐）

配置会在加载国家配置时自动读取：

```python
from ner.config import ConfigManager
from ner.postprocess import build_postprocessor_from_config

# 加载配置
config = ConfigManager().load_country_config('uae')

# 构建后处理器
postprocessor = build_postprocessor_from_config(config)

# 使用
if postprocessor:
    tokens, labels, confidences = postprocessor.apply(tokens, labels, confidences)
```

### 在代码中集成

#### 1. 在 evaluator 中

```python
from ner.evaluation import NEREvaluator
from ner.postprocess import build_postprocessor_from_config

# 构建后处理器
postprocessor = build_postprocessor_from_config(config)

# 创建评估器
evaluator = NEREvaluator(
    model, 
    tokenizer, 
    label_list,
    postprocessor=postprocessor  # 传入后处理器
)

# 评估时自动应用
results = evaluator.evaluate_dataloader(dataloader)
```

#### 2. 在 predict 中

```python
# 预测时传入后处理器
prediction = model.predict(
    text,
    tokenizer=tokenizer,
    postprocessor=postprocessor
)
```

#### 3. 在 REPL 中

```python
# 加载模型时自动加载后处理配置
class PredictREPL:
    def _handle_load(self, model_path: str):
        # ... 加载模型 ...
        
        # 加载后处理配置
        config_path = Path(model_path) / "postprocess_config.json"
        if config_path.exists():
            self.postprocessor = build_postprocessor_from_file(str(config_path))
```

## 📚 可用规则

| 规则类型 | 功能 | 推荐场景 |
|---------|------|---------|
| `bio_consistency` | BIO标签一致性 | 所有场景（必选） |
| `confidence_threshold` | 置信度过滤 | 高精度场景 |
| `entity_boundary` | 边界修正 | 所有场景（推荐） |
| `min_entity_length` | 最小长度过滤 | 减少误识别 |
| `pattern_correction` | 模式匹配修正 | 格式化实体 |
| `merge_adjacent` | 合并相邻实体 | 高召回场景 |

详细参数说明见：`docs/POSTPROCESS_CONFIG.md`

## 🎨 配置模板

### 基础模板（适用于大多数场景）

```json
{
  "postprocess": {
    "rules": [
      {"type": "bio_consistency", "params": {"fix_orphan_i": "to_b"}},
      {"type": "entity_boundary", "params": {"remove_boundary_punct": true}}
    ]
  }
}
```

### 严格模板（高精度）

```json
{
  "postprocess": {
    "rules": [
      {"type": "bio_consistency", "params": {"fix_orphan_i": "to_b"}},
      {"type": "confidence_threshold", "params": {"threshold": 0.7}},
      {"type": "entity_boundary", "params": {"remove_boundary_punct": true}},
      {"type": "min_entity_length", "params": {"min_length": 2}}
    ]
  }
}
```

### 宽松模板（高召回）

```json
{
  "postprocess": {
    "rules": [
      {"type": "bio_consistency", "params": {"fix_orphan_i": "to_b"}},
      {"type": "merge_adjacent", "params": {"max_gap": 2}}
    ]
  }
}
```

## ✅ 验证配置

运行示例脚本验证配置：

```bash
python examples/postprocess_config_example.py
```

或在Python中：

```python
from ner.config import ConfigManager
from ner.postprocess import build_postprocessor_from_config

config = ConfigManager().load_country_config('uae')
postprocessor = build_postprocessor_from_config(config)

if postprocessor:
    print(f"✓ 成功加载 {len(postprocessor.rules)} 个规则")
    for rule in postprocessor.rules:
        print(f"  - {rule.name}")
else:
    print("✗ 未配置后处理规则")
```

## 🐛 常见问题

### Q: 配置后不生效？

A: 检查以下几点：
1. 配置文件格式是否正确（JSON语法）
2. 规则类型名称是否正确
3. 是否正确传入了 `postprocessor` 参数
4. 查看日志确认是否加载成功

### Q: 如何禁用后处理？

A: 三种方式：
1. 删除配置文件中的 `postprocess` 部分
2. 将 `rules` 设为空数组：`"rules": []`
3. 代码中不传入 `postprocessor` 参数

### Q: 如何调试规则效果？

A: 
1. 使用示例脚本测试：`examples/postprocess_example.py`
2. 一次只启用一个规则，观察效果
3. 启用详细日志查看规则应用过程
4. 对比有无后处理的预测结果

### Q: 规则顺序重要吗？

A: 是的！规则按配置顺序依次应用。建议顺序：
1. `bio_consistency`（修正基础错误）
2. `confidence_threshold`（过滤低置信度）
3. `entity_boundary`（修正边界）
4. `min_entity_length`（过滤短实体）
5. 其他规则

## 📖 更多文档

- 完整配置说明：`docs/POSTPROCESS_CONFIG.md`
- 设计文档：`docs/POSTPROCESS_DESIGN.md`
- 快速参考：`docs/POSTPROCESS_QUICKREF.md`
- 集成指南：`POSTPROCESS_INTEGRATION.md`
