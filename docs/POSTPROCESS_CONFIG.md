# 后处理器配置指南

## 配置位置

后处理规则配置在国家配置文件中：

```
data/ner/configs/countries/{country}.json
```

例如：`data/ner/configs/countries/uae.json`

## 配置结构

在国家配置文件中添加 `postprocess` 部分：

```json
{
  "country": { ... },
  "model": { ... },
  "training": { ... },
  "data": { ... },
  "labels": { ... },
  "evaluation": { ... },
  "postprocess": {
    "rules": [
      {
        "type": "规则类型",
        "params": {
          "参数名": "参数值"
        }
      }
    ]
  },
  "output": { ... }
}
```

## 可用规则及参数

### 1. BIO一致性规则 (bio_consistency)

确保BIO标签序列的一致性。

```json
{
  "type": "bio_consistency",
  "params": {
    "fix_orphan_i": "to_b"
  }
}
```

**参数说明：**
- `fix_orphan_i`: 如何处理孤立的I标签
  - `"to_b"`: 转换为B标签（默认，推荐）
  - `"to_o"`: 转换为O标签

**适用场景：**
- 所有场景（强烈推荐）
- 修正模型预测的BIO序列错误

---

### 2. 置信度阈值规则 (confidence_threshold)

过滤低置信度的预测。

```json
{
  "type": "confidence_threshold",
  "params": {
    "threshold": 0.5,
    "keep_entity_if_any_high": false
  }
}
```

**参数说明：**
- `threshold`: 置信度阈值（0.0-1.0）
  - 默认：0.5
  - 建议：0.5-0.7（根据场景调整）
- `keep_entity_if_any_high`: 实体级过滤策略
  - `false`: 逐token过滤（默认）
  - `true`: 如果实体中任何token高于阈值，保留整个实体

**适用场景：**
- 需要高精度的场景
- 减少误报（False Positives）

---

### 3. 实体边界规则 (entity_boundary)

修正实体边界问题，移除边界的标点和停用词。

```json
{
  "type": "entity_boundary",
  "params": {
    "remove_boundary_punct": true,
    "remove_boundary_stopwords": false,
    "custom_punct": null,
    "custom_stopwords": null
  }
}
```

**参数说明：**
- `remove_boundary_punct`: 是否移除边界标点
  - 默认：true（推荐）
- `remove_boundary_stopwords`: 是否移除边界停用词
  - 默认：false
- `custom_punct`: 自定义标点集合（可选）
  - 例如：`[".", ",", "!", "?"]`
- `custom_stopwords`: 自定义停用词集合（可选）
  - 例如：`["the", "a", "an"]`

**适用场景：**
- 所有场景（推荐）
- 提升实体边界质量

---

### 4. 最小实体长度规则 (min_entity_length)

过滤过短的实体。

```json
{
  "type": "min_entity_length",
  "params": {
    "min_length": 2,
    "min_length_by_type": {
      "BUILDING": 2,
      "STREET": 2,
      "PERSON": 2
    }
  }
}
```

**参数说明：**
- `min_length`: 默认最小长度（token数）
  - 默认：2
  - 建议：1-3
- `min_length_by_type`: 按实体类型指定最小长度
  - 可选，覆盖默认值

**适用场景：**
- 减少单字符误识别
- 特定实体类型有最小长度要求

---

### 5. 模式修正规则 (pattern_correction)

基于正则表达式模式修正实体类型。

```json
{
  "type": "pattern_correction",
  "params": {
    "patterns": {
      "PHONE": "\\+?\\d{1,3}[-\\.\\s]?\\d{3,}",
      "EMAIL": "\\S+@\\S+\\.\\S+",
      "POSTCODE": "\\d{5,6}"
    }
  }
}
```

**参数说明：**
- `patterns`: 实体类型到正则表达式的映射
  - 键：实体类型名
  - 值：正则表达式（注意转义）

**适用场景：**
- 有明确格式的实体（电话、邮箱、邮编等）
- 修正模型对格式化实体的误判

---

### 6. 合并相邻实体规则 (merge_adjacent)

合并相邻的同类型实体。

```json
{
  "type": "merge_adjacent",
  "params": {
    "merge_across_tokens": [" ", "-", "/", "&"],
    "max_gap": 1
  }
}
```

**参数说明：**
- `merge_across_tokens`: 允许跨越的token集合
  - 默认：`[" ", "-", "/", "&"]`
- `max_gap`: 最大允许的间隔token数
  - 默认：1

**适用场景：**
- 实体被分隔符分割的情况
- 提高召回率

---

## 完整配置示例

### 基础配置（推荐）

适用于大多数场景：

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
      }
    ]
  }
}
```

### 严格配置（高精度）

适用于需要高精度的场景：

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
        "type": "confidence_threshold",
        "params": {
          "threshold": 0.7,
          "keep_entity_if_any_high": false
        }
      },
      {
        "type": "entity_boundary",
        "params": {
          "remove_boundary_punct": true,
          "remove_boundary_stopwords": true
        }
      },
      {
        "type": "min_entity_length",
        "params": {
          "min_length": 2
        }
      }
    ]
  }
}
```

### 宽松配置（高召回）

适用于需要高召回率的场景：

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
        "type": "merge_adjacent",
        "params": {
          "merge_across_tokens": [" ", "-", "/", "&", ","],
          "max_gap": 2
        }
      }
    ]
  }
}
```

### 特定领域配置（地址识别）

针对地址实体的配置：

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
          "custom_punct": [".", ",", "!", "?", ";", ":"]
        }
      },
      {
        "type": "min_entity_length",
        "params": {
          "min_length": 1,
          "min_length_by_type": {
            "BUILDING": 2,
            "STREET": 2,
            "COMMUNITY": 2,
            "EMIRATE": 1,
            "COUNTRY": 1
          }
        }
      },
      {
        "type": "pattern_correction",
        "params": {
          "patterns": {
            "POSTCODE": "\\d{5,6}"
          }
        }
      }
    ]
  }
}
```

## 配置加载方式

### 方式1：通过ConfigManager自动加载（推荐）

配置会在加载国家配置时自动读取：

```python
from ner.config import ConfigManager
from ner.postprocess import build_postprocessor_from_config

config_manager = ConfigManager()
config = config_manager.load_country_config('uae')

# 构建后处理器
postprocessor = build_postprocessor_from_config(config)
```

### 方式2：直接从文件加载

```python
from ner.postprocess import build_postprocessor_from_file

postprocessor = build_postprocessor_from_file('data/ner/configs/countries/uae.json')
```

### 方式3：程序化配置

```python
from ner.postprocess import Postprocessor
from ner.postprocess.rules import BIOConsistencyRule, EntityBoundaryRule

postprocessor = Postprocessor([
    BIOConsistencyRule(fix_orphan_i='to_b'),
    EntityBoundaryRule(remove_boundary_punct=True)
])
```

## 配置验证

配置加载时会自动验证：
- 规则类型是否存在
- 参数是否有效
- 数据类型是否正确

如果配置有误，会抛出明确的错误信息。

## 调试建议

1. **从简单开始**：先只配置 `bio_consistency`
2. **逐步添加**：一次添加一个规则，观察效果
3. **查看日志**：启用详细日志查看规则应用情况
4. **A/B测试**：对比有无后处理的效果差异

## 性能考虑

- 规则按顺序执行，将快速规则放前面
- `bio_consistency` 和 `entity_boundary` 开销小，推荐使用
- `pattern_correction` 涉及正则匹配，模式复杂时可能较慢
- 批量处理时性能影响可忽略
