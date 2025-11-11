# 后处理器模块 - 完整指南

## 📋 概述

后处理器模块为NER预测结果提供规则化修正能力，在模型预测后应用规则来提升结果质量。

## 🎯 核心特性

- ✅ **通用性**：统一接口，适用于evaluate和predict场景
- ✅ **可组合**：多个规则链式应用
- ✅ **配置驱动**：从JSON配置文件加载
- ✅ **易扩展**：继承BaseRule即可添加自定义规则
- ✅ **性能友好**：支持批量处理

## 📍 配置位置

后处理规则配置在国家配置文件中：

```
data/ner/configs/countries/{country}.json
```

例如：`data/ner/configs/countries/uae.json`（已配置）

## 🚀 快速开始

### 1. 查看现有配置

UAE配置文件已包含后处理规则：

```json
{
  "postprocess": {
    "rules": [
      {
        "type": "bio_consistency",
        "params": {"fix_orphan_i": "to_b"}
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

### 2. 在代码中使用

```python
from ner.config import ConfigManager
from ner.postprocess import build_postprocessor_from_config

# 加载配置
config = ConfigManager().load_country_config('uae')

# 构建后处理器
postprocessor = build_postprocessor_from_config(config)

# 使用
if postprocessor:
    tokens, labels, confidences = postprocessor.apply(
        tokens, labels, confidences
    )
```

### 3. 集成到评估和预测

#### 在 evaluator 中

```python
from ner.evaluation import NEREvaluator

evaluator = NEREvaluator(
    model, 
    tokenizer, 
    label_list,
    postprocessor=postprocessor  # 传入后处理器
)

results = evaluator.evaluate_dataloader(dataloader)
```

#### 在 predict 中

```python
prediction = model.predict(
    text,
    tokenizer=tokenizer,
    postprocessor=postprocessor  # 传入后处理器
)
```

## 📦 模块结构

```
src/ner/postprocess/
├── __init__.py           # 公共API
├── pipeline.py           # 核心类
├── rules.py              # 6个内置规则
├── builder.py            # 配置构建器
└── README.md             # 使用文档

docs/
├── POSTPROCESS_CONFIG.md    # 完整配置说明
├── POSTPROCESS_DESIGN.md    # 设计文档
├── POSTPROCESS_QUICKREF.md  # 快速参考
└── POSTPROCESS_SUMMARY.md   # 配置总结

examples/
├── postprocess_example.py        # 使用示例
└── postprocess_config_example.py # 配置示例

tests/
└── test_postprocess.py          # 单元测试
```

## 🎨 内置规则

| 规则 | 功能 | 推荐场景 |
|------|------|---------|
| `bio_consistency` | BIO标签一致性修正 | 所有场景（必选） |
| `confidence_threshold` | 置信度阈值过滤 | 高精度场景 |
| `entity_boundary` | 实体边界修正 | 所有场景（推荐） |
| `min_entity_length` | 最小实体长度过滤 | 减少误识别 |
| `pattern_correction` | 基于正则模式修正 | 格式化实体 |
| `merge_adjacent` | 合并相邻同类实体 | 高召回场景 |

## 📖 文档导航

### 快速入门
- **配置总结**：`docs/POSTPROCESS_SUMMARY.md` ⭐ 推荐先看
- **快速参考**：`docs/POSTPROCESS_QUICKREF.md`

### 详细文档
- **配置说明**：`docs/POSTPROCESS_CONFIG.md` - 所有规则的详细参数
- **设计文档**：`docs/POSTPROCESS_DESIGN.md` - 架构和设计理念
- **集成指南**：`POSTPROCESS_INTEGRATION.md` - 如何集成到现有代码

### 示例代码
- **基础示例**：`examples/postprocess_example.py` - 6个使用示例
- **配置示例**：`examples/postprocess_config_example.py` - 配置加载示例

### 测试
- **单元测试**：`tests/test_postprocess.py` - 运行测试验证功能

## 🔧 常用配置模板

### 基础配置（推荐）

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

### 严格配置（高精度）

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

### 宽松配置（高召回）

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

运行示例脚本验证配置是否正确：

```bash
# 基础示例
python examples/postprocess_example.py

# 配置加载示例
python examples/postprocess_config_example.py

# 运行测试
python -m pytest tests/test_postprocess.py -v
```

## 🎓 使用建议

### 推荐规则组合

1. **所有场景必选**：`bio_consistency`
2. **大多数场景推荐**：`entity_boundary`
3. **根据需求选择**：
   - 高精度 → 添加 `confidence_threshold`
   - 减少误识别 → 添加 `min_entity_length`
   - 特定格式实体 → 添加 `pattern_correction`
   - 提高召回 → 添加 `merge_adjacent`

### 规则顺序建议

```
1. bio_consistency      # 修正基础BIO错误
2. confidence_threshold # 过滤低置信度
3. entity_boundary      # 修正边界
4. min_entity_length    # 过滤短实体
5. 其他规则
```

### 调试技巧

1. 从最简单的配置开始（只用 `bio_consistency`）
2. 一次添加一个规则，观察效果
3. 使用示例脚本测试规则效果
4. 对比有无后处理的预测结果
5. 根据实际效果调整参数

## 🐛 常见问题

### Q: 配置后不生效？

检查：
1. JSON格式是否正确
2. 规则类型名称是否正确
3. 是否传入了 `postprocessor` 参数
4. 查看日志确认加载状态

### Q: 如何禁用后处理？

三种方式：
1. 删除配置中的 `postprocess` 部分
2. 设置 `"rules": []`
3. 代码中不传入 `postprocessor` 参数

### Q: 如何添加自定义规则？

```python
from ner.postprocess import BaseRule, PredictionResult

class MyRule(BaseRule):
    name = 'my_rule'
    
    def apply(self, result: PredictionResult) -> PredictionResult:
        # 实现逻辑
        return result

# 注册
from ner.postprocess.builder import RULE_FACTORY
RULE_FACTORY['my_rule'] = MyRule
```

## 📞 获取帮助

- 查看文档：`docs/POSTPROCESS_*.md`
- 运行示例：`examples/postprocess_*.py`
- 查看测试：`tests/test_postprocess.py`

## 🎉 总结

后处理器模块已完全集成到NER系统中：

✅ 模块已创建并实现  
✅ UAE配置已添加后处理规则  
✅ 文档已完善  
✅ 示例代码已提供  
✅ 测试已编写  

现在你可以：
1. 直接使用UAE配置中的后处理规则
2. 根据需求修改配置
3. 在evaluate和predict中自动应用
4. 添加自定义规则扩展功能

开始使用吧！🚀
