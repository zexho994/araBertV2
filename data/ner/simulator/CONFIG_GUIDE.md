# 配置文件详细说明

本文档详细解释 `generator_config.json` 中每个配置项的含义和使用方法。

## 📋 完整配置结构

```json5
{
    "generate_size": 100,                    // 生成地址的总数量
    "unique_combinations_only": true,        // 是否只保留唯一组合
    
    "dictionaries": { ... },                 // 实体词典路径
    "templates": { ... },                    // 地址模板和分隔符配置
    "typo_injection": { ... },               // 拼写错误和大小写变体配置
    "noise_injection": { ... },              // 噪音注入配置
    "output": { ... }                        // 输出配置
}
```

---

## 1. 基础配置

### `generate_size`
- **类型**: 整数
- **默认值**: `100`
- **说明**: 要生成的地址总数量
- **示例**:
  ```json5
  "generate_size": 1000  // 生成1000条地址
  ```

### `unique_combinations_only`
- **类型**: 布尔值
- **默认值**: `true`
- **说明**: 是否去重（只保留唯一的地址组合）
- **注意**: 启用去重时，如果词典实体数量较少，可能无法生成足够多的唯一组合

---

## 2. 词典配置 (`dictionaries`)

指定各实体类型的词典文件路径（相对于项目根目录）。

```json5
"dictionaries": {
    "COUNTRY": "data/ner/simulator/uae/dictionaries/country.txt",
    "EMIRATE": "data/ner/simulator/uae/dictionaries/emirate.txt",
    "CITY": "data/ner/simulator/uae/dictionaries/city.txt",
    "SUB_AREA": "data/ner/simulator/uae/dictionaries/sub_area.txt",
    "COMPOUND": "data/ner/simulator/uae/dictionaries/compound.txt",
    "STREET": "data/ner/simulator/uae/dictionaries/street.txt",
    "BUILDING": "data/ner/simulator/uae/dictionaries/building.txt"
}
```

**词典文件格式**:
- UTF-8 编码
- 每行一个实体
- 空行会被忽略

---

## 3. 模板配置 (`templates`)

### 3.1 模板定义 (`definitions`)

定义地址的组合模式和权重。

```json5
"definitions": [
    {
        "pattern": "{BUILDING} {SUB_AREA} {CITY} {COUNTRY}",
        "weight": 0.25  // 权重 25%
    },
    {
        "pattern": "{BUILDING} {STREET} {SUB_AREA} {CITY} {COUNTRY}",
        "weight": 0.20  // 权重 20%
    }
]
```

**说明**:
- `pattern`: 地址模板，使用 `{实体类型}` 作为占位符
- `weight`: 权重，所有模板权重之和应为 1.0
- 权重越大，该模板被选中的概率越高

**支持的实体类型**:
- `{BUILDING}` - 建筑物
- `{STREET}` - 街道
- `{COMPOUND}` - 社区/小区
- `{SUB_AREA}` - 子区域
- `{CITY}` - 城市
- `{EMIRATE}` - 酋长国（UAE特有）
- `{COUNTRY}` - 国家

### 3.2 分隔符变体 (`separator_variations`)

```json5
"separator_variations": {
    "options": [",", "/", "-"],    // 可选分隔符
    "probability": 0.6             // 添加分隔符的概率
}
```

**说明**:
- `options`: 可选的分隔符列表
- `probability`: 为地址添加分隔符的概率（0.0-1.0）
- 分隔符会被添加在实体之间

**示例**:
```
无分隔符: Villa 276 Dubai UAE
有分隔符: Villa 276, Dubai, UAE
```

---

## 4. 拼写错误配置 (`typo_injection`)

### 4.1 全局配置

```json5
"typo_injection": {
    "enabled": true,              // 是否启用拼写错误注入
    "global_probability": 0.8,    // 地址进行typo处理的概率
    ...
}
```

### 4.2 大小写变体 (`case_variations`)

```json5
"case_variations": {
    "enabled": true,              // 是否启用大小写变体
    "probability": 0.4,           // 应用变体的概率
    "strategies": {
        "normal_case": 0.70,      // 保持原样: 70%
        "all_lowercase": 0.15,    // 全小写: 15%
        "all_uppercase": 0.10,    // 全大写: 10%
        "random_case": 0.05       // 随机大小写: 5%
    }
}
```

**示例**:
```
normal_case:    Villa 276 Dubai UAE
all_lowercase:  villa 276 dubai uae
all_uppercase:  VILLA 276 DUBAI UAE
random_case:    ViLLa 276 dUbAi uAe
```

### 4.3 实体级拼写错误 (`per_entity_config`)

为每种实体类型配置不同的拼写错误概率和类型。

```json5
"BUILDING": {
    "probability": 0.6,           // 该实体发生拼写错误的概率
    "max_char_error_ratio": 0.15, // 最大字符错误比例
    "typo_types": {
        "swap": 0.4,              // 交换相邻字符: 40%
        "deletion": 0.3,          // 删除字符: 30%
        "insertion": 0.2,         // 插入字符: 20%
        "substitution": 0.1       // 替换字符: 10%
    }
}
```

**拼写错误类型示例**:
```
原始:        Street
swap:        Steret  (e和r交换)
deletion:    Stret   (删除e)
insertion:   Streeat (插入e)
substitution: Strext (e替换为x)
```

**建议配置**:
- 重要实体（CITY, EMIRATE, COUNTRY）: 低概率 (0.01-0.02)
- 次要实体（SUB_AREA, COMPOUND）: 中等概率 (0.02-0.05)
- 细节实体（BUILDING, STREET）: 高概率 (0.05-0.10)

---

## 5. 噪音注入配置 (`noise_injection`)

在实体之间添加噪音，模拟真实世界的地址数据。

### 5.1 全局配置

```json5
"noise_injection": {
    "enabled": true,                  // 是否启用噪音注入
    "global_noise_probability": 0.8,  // 地址注入噪音的概率
    "max_noise_per_address": 3,       // 每个地址最多注入噪音数
    ...
}
```

### 5.2 标点符号噪音 (`punctuation`)

```json5
"punctuation": {
    "enabled": true,
    "probability": 0.35,                        // 该类型噪音的概率
    "characters": [".", ",", ";", "-", "/"],    // 可选字符
    "max_consecutive": 3                        // 最多连续个数
}
```

**示例**:
```
Villa 276, Dubai UAE  →  Villa 276,, Dubai UAE
Villa 276 Dubai UAE   →  Villa 276 ./: Dubai UAE
```

### 5.3 数字噪音 (`numbers`)

```json5
"numbers": {
    "enabled": true,
    "probability": 0.25,
    "patterns": {
        "random_digits": {
            "min_length": 1,
            "max_length": 4,
            "probability": 0.6      // 随机数字: 60%
        },
        "phone_like": {
            "format": "+971XXXXXXXXX",
            "probability": 0.25     // 电话号码格式: 25%
        },
        "po_box": {
            "format": "P.O. Box XXXXX",
            "probability": 0.15     // P.O. Box格式: 15%
        }
    }
}
```

**示例**:
```
random_digits:  Villa 276 1234 Dubai UAE
phone_like:     Villa 276 +971501234567 Dubai UAE
po_box:         Villa 276 P.O. Box 12345 Dubai UAE
```

### 5.4 无意义词噪音 (`meaningless_words`)

```json5
"meaningless_words": {
    "enabled": true,
    "probability": 0.4,
    "word_list": [
        "Near", "Opposite", "Behind", "Next to",
        "بجانب", "قريب من", "مقابل",
        "Floor", "Unit", "Block"
    ],
    "max_per_address": 2
}
```

**示例**:
```
Villa 276 Dubai UAE  →  Villa 276 Near Dubai UAE
Villa 276 Dubai UAE  →  Villa 276 Floor 5 Dubai UAE
```

---

## 6. 输出配置 (`output`)

### 6.1 基础配置

```json5
"output": {
    "format": "csv",                          // 输出格式
    "output_dir": "../raw_data/",             // 输出目录（相对于配置文件）
    "file_prefix": "simulated_",              // 文件名前缀
    "timestamp_suffix": true,                 // 是否添加时间戳后缀
    "fields": [                               // 输出字段（按顺序）
        "formatted_address",
        "BUILDING",
        "STREET",
        "COMPOUND",
        "SUB_AREA",
        "CITY",
        "EMIRATE",
        "COUNTRY"
    ],
    ...
}
```

**输出文件示例**:
```
simulated_20251010_110225.csv
```

### 6.2 数据处理

```json5
"deduplicate": true,      // 是否去重
"shuffle": false,         // 是否打乱顺序
"include_metadata": false // 是否包含元数据（生成时间等）
```

### 6.3 质量验证

```json5
"quality_validation": {
    "enabled": true,
    "min_address_length": 20,    // 最小地址长度
    "max_address_length": 500    // 最大地址长度
}
```

---

## 📝 配置示例

### 示例1: 生成高质量训练数据

```json5
{
    "generate_size": 5000,
    "unique_combinations_only": true,
    
    "typo_injection": {
        "enabled": false,  // 不添加拼写错误
        "case_variations": {
            "enabled": true,
            "probability": 0.3  // 减少大小写变体
        }
    },
    
    "noise_injection": {
        "enabled": true,
        "global_noise_probability": 0.5,  // 减少噪音
        "max_noise_per_address": 1
    }
}
```

### 示例2: 生成高噪音鲁棒性数据

```json5
{
    "generate_size": 2000,
    "unique_combinations_only": false,  // 允许重复
    
    "typo_injection": {
        "enabled": true,
        "global_probability": 0.9,  // 提高拼写错误概率
        "case_variations": {
            "probability": 0.6  // 提高变体概率
        }
    },
    
    "noise_injection": {
        "enabled": true,
        "global_noise_probability": 0.9,
        "max_noise_per_address": 5  // 增加噪音数量
    }
}
```

### 示例3: 生成测试数据（简洁清晰）

```json5
{
    "generate_size": 100,
    "unique_combinations_only": true,
    
    "templates": {
        "separator_variations": {
            "probability": 0.9  // 总是添加分隔符
        }
    },
    
    "typo_injection": {
        "enabled": false  // 不添加拼写错误
    },
    
    "noise_injection": {
        "enabled": false  // 不添加噪音
    }
}
```

---

## 🔧 调优建议

### 1. 提高数据多样性
- 增加模板数量和权重分布
- 提高噪音注入概率
- 增加词典实体数量

### 2. 提高数据质量
- 降低拼写错误概率
- 减少噪音注入
- 启用质量验证

### 3. 平衡真实性
- 为重要实体（CITY, COUNTRY）设置低拼写错误率
- 为细节实体（BUILDING）设置中等拼写错误率
- 适度添加噪音（40%-60%概率）

### 4. 生成特定用途数据

**清洁数据（用于预训练）**:
```json5
typo_injection.enabled: false
noise_injection.global_noise_probability: 0.3
```

**噪音数据（用于鲁棒性训练）**:
```json5
typo_injection.global_probability: 0.9
noise_injection.global_noise_probability: 0.9
```

**混合数据（推荐）**:
```json5
typo_injection.global_probability: 0.5
noise_injection.global_noise_probability: 0.6
```

---

## ⚠️ 常见问题

**Q: 为什么生成的数量少于 generate_size？**

A: 可能原因：
1. 启用了 `unique_combinations_only`，但词典组合空间不足
2. 质量验证过滤了太多地址

解决方案：
- 关闭 `unique_combinations_only`
- 增加词典实体数量
- 调整质量验证参数

**Q: 如何控制生成速度？**

A: 
- 降低 `typo_injection.global_probability`
- 降低 `noise_injection.global_noise_probability`
- 减少噪音类型

**Q: 生成的地址不够真实？**

A: 
- 检查词典质量
- 调整模板使其更符合真实地址格式
- 适度降低噪音概率

---

## 📚 参考

详细使用指南: [README_USAGE.md](README_USAGE.md)

主README: [README.md](README.md)

