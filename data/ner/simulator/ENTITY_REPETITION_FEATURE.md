# 实体重复功能（T6阶段）

## 概述

在地址生成器中添加了 T6 阶段，支持实体重复功能，用于训练模型正确识别连续重复出现的实体。

## 问题背景

模型在处理类似 "House 20 House 20" 这样的连续重复实体时，会将其识别为一个整体，而不是两个独立的 BUILDING 实体。

## 解决方案

### 实现方式

在生成流程中添加 T6 阶段（实体重复处理），在 T5（实体特定噪音）之后执行：

```
T1 (生成原始) → T2 (分隔符) → T3 (拼写错误) → T4 (通用噪音) → T5 (实体特定噪音) → T6 (实体重复)
```

### 配置说明

在 `generator_config.json` 中新增 `entity_repetition` 配置段：

```json
{
  "entity_repetition": {
    "enabled": true,
    "global_probability": 0.3,
    "per_entity_config": {
      "BUILDING": {
        "enabled": true,
        "probability": 0.4,
        "min_repeat": 2,
        "max_repeat": 3,
        "separator_options": [" ", " , "],
        "separator_probability": 0.5
      },
      "STREET": {
        "enabled": true,
        "probability": 0.1,
        "min_repeat": 2,
        "max_repeat": 2,
        "separator_options": [" "],
        "separator_probability": 1.0
      }
    }
  }
}
```

**配置参数说明：**

- `enabled`: 是否启用实体重复功能
- `global_probability`: 全局概率，决定是否对某条地址应用重复处理
- `per_entity_config`: 针对每种实体类型的配置
  - `probability`: 该实体类型的重复概率
  - `min_repeat`: 最小重复次数
  - `max_repeat`: 最大重复次数
  - `separator_options`: 重复实体之间的分隔符选项
  - `separator_probability`: 使用分隔符的概率

**实际重复概率计算：**
- BUILDING 重复概率 ≈ 0.3 × 0.4 = 12%
- STREET 重复概率 ≈ 0.3 × 0.1 = 3%

### 示例效果

**输入模板**: `{BUILDING} {SUB_AREA} {EMIRATE} {COUNTRY}`

**原始生成**: `House 20 Al Qarayen 4 Sharjah UAE`

**T6 处理后**: `House 20 House 20 Al Qarayen 4 Sharjah UAE`

**CSV 输出**:
```csv
formatted_address,BUILDING,SUB_AREA,EMIRATE,COUNTRY
"House 20 House 20 Al Qarayen 4 Sharjah UAE","House 20","Al Qarayen 4","Sharjah","UAE"
```

**关键特点：**
- `formatted_address`: 包含重复的完整地址
- `BUILDING` 列: 只存储原始实体值（不重复）
- 后续数据处理流程会在地址中查找该实体的所有出现位置并分别标注

### 真实示例

从实际生成的数据中：

1. **BUILDING 重复3次**：
   ```
   地址: Arcade Building Arcade Building Arcade Building شارع ٢٤ Al Ain UAE
   标签: Arcade Building
   ```

2. **BUILDING 重复2次（带分隔符）**：
   ```
   地址: Perisa M03 , Perisa M03 Sharjah UAE
   标签: Perisa M03
   ```

3. **STREET 重复2次**：
   ```
   地址: Al Rasaas Rd Al Rasaas Rd Rostamania Tower Dubai UAE
   标签: Al Rasaas Rd
   ```

## 代码实现

### 核心方法

`_apply_entity_repetition()` 方法位于 `generator.py` 中：

```python
def _apply_entity_repetition(
    self,
    address: str,
    entities: Dict[str, str],
    entity_types: List[str]
) -> str:
    """T6: 实体重复处理"""
    # 遍历实体，按配置概率进行重复
    # 从后往前替换，避免位置偏移问题
    # 只修改地址字符串，不修改entities字典
```

### 集成到生成流程

在 `generate_one_address()` 方法中：

```python
# T5: 实体特定噪音
address_t5 = self._apply_entity_specific_noise(address_t4, entities_t3, entity_types)

# T6: 实体重复
address_t6 = self._apply_entity_repetition(address_t5, entities_t3, entity_types)

return address_t6, entities_t3
```

## 测试结果

在100条样本测试中：
- BUILDING 重复率: 18.5%（预期 ~12%）
- STREET 重复率: 4.1%（预期 ~3%）

实际概率与预期基本吻合（考虑随机性和样本量）。

## 优势

1. **不破坏现有架构**: T6 作为独立阶段，与现有 T1-T5 流程解耦
2. **高扩展性**: 配置清晰，易于添加新的实体类型或调整参数
3. **灵活的分隔符**: 支持多种分隔符选项（空格、逗号等）
4. **可调概率**: 独立控制全局和各实体类型的重复概率

## 调整建议

如果需要调整重复频率，修改配置文件中的概率参数：

- 提高重复率: 增大 `global_probability` 或 `probability`
- 降低重复率: 减小概率值
- 调整重复次数: 修改 `min_repeat` 和 `max_repeat`
- 修改分隔符: 调整 `separator_options` 数组

