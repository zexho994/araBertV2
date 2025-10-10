# 地址数据模拟器使用指南

## 1. 运行生成器

使用默认配置
```shell
python data/ner/simulator/generator.py
```

或指定配置文件
```
python data/ner/simulator/generator.py data/ner/simulator/uae/config/generator_config.json5
```

## 2. 生成流程说明

生成器按照以下流程生成地址：

### T1: 生成原始地址
- 从模板定义中按权重随机选择一个模板
- 从对应实体词典中随机选择实体值
- 填充模板生成原始地址

**示例**:
```
模板: {BUILDING} {SUB_AREA} {CITY} {COUNTRY}
↓
Villa 276 Al Yalayis 4 Dubai UAE
```

### T2: 添加分隔符
- 按照配置的概率在实体间添加分隔符
- 分隔符从 options 中随机选择（`,`, `/`, `-`）

**示例**:
```
Villa 276 Al Yalayis 4 Dubai UAE
↓
Villa 276, Al Yalayis 4, Dubai, UAE
```

### T3: 应用大小写变体
- 按照配置的概率对地址应用大小写变体
- 支持4种策略：正常、全小写、全大写、随机大小写

**示例**:
```
Villa 276, Al Yalayis 4, Dubai, UAE
↓
villa 276, al yalayis 4, dubai, uae  (全小写)
```

### T4: 注入噪音
- 在实体之间注入噪音（不破坏实体本身）
- 支持3种噪音类型：
  - 标点符号
  - 数字（随机数字、电话号码、P.O. Box）
  - 无意义词（Near, Opposite, Floor, etc.）

**示例**:
```
villa 276, al yalayis 4, dubai, uae
↓
villa 276, Near al yalayis 4, 1234 dubai, uae
```

### 3. 配置调整

编辑 `data/ner/simulator/uae/config/generator_config.json5` 来调整生成参数：

#### 控制生成数量
```json
{
    "generate_size": 100,  // 修改此值
    "unique_combinations_only": true  // 是否去重
}
```

#### 调整模板权重
```json
{
    "templates": {
        "definitions": [
            {
                "pattern": "{BUILDING} {SUB_AREA} {CITY} {COUNTRY}",
                "weight": 0.25  // 权重越大，被选中概率越高
            }
        ]
    }
}
```

#### 控制分隔符
```json
{
    "separator_variations": {
        "options": [",", "/", "-"],  // 可选分隔符
        "probability": 0.6  // 添加分隔符的概率
    }
}
```

#### 控制大小写变体
```json
{
    "case_variations": {
        "enabled": true,
        "probability": 0.4,  // 应用变体的概率
        "strategies": {
            "normal_case": 0.70,    // 保持原样: 70%
            "all_lowercase": 0.15,  // 全小写: 15%
            "all_uppercase": 0.10,  // 全大写: 10%
            "random_case": 0.05     // 随机: 5%
        }
    }
}
```

#### 控制噪音注入
```json
{
    "noise_injection": {
        "enabled": true,
        "global_noise_probability": 0.8,  // 地址注入噪音的概率
        "max_noise_per_address": 3,       // 每个地址最多注入噪音数
        "noise_types": {
            "punctuation": {
                "probability": 0.35  // 该类型噪音的概率
            },
            "numbers": {
                "probability": 0.25
            },
            "meaningless_words": {
                "probability": 0.4
            }
        }
    }
}
```

### 4. 输出说明

生成的CSV文件包含以下字段：
- `formatted_address`: 完整的生成地址
- `BUILDING`: 建筑物实体
- `STREET`: 街道实体
- `COMPOUND`: 社区实体
- `SUB_AREA`: 子区域实体
- `CITY`: 城市实体
- `EMIRATE`: 酋长国实体
- `COUNTRY`: 国家实体

输出文件位置：
```
data/ner/raw_data/simulated_YYYYMMDD_HHMMSS.csv
```

### 5. 质量控制

生成器自动进行以下质量控制：
- ✅ 地址长度验证（20-500字符）
- ✅ 去重（如果启用）
- ✅ 格式验证

### 6. 常见问题

**Q: 生成的地址数量少于目标数量？**

A: 可能原因：
- 词典实体数量太少，组合空间不足
- `unique_combinations_only` 启用时，无法生成足够多的唯一组合
- 质量验证过滤了太多地址

解决方案：
- 增加词典实体数量
- 关闭唯一性要求
- 调整质量验证参数

**Q: 如何增加地址的多样性？**

A: 
- 增加更多模板
- 提高噪音注入概率
- 添加更多词典实体
- 增加分隔符变体

**Q: 生成的地址看起来不真实？**

A: 
- 降低噪音注入概率
- 减少大小写变体概率
- 使用更合理的模板组合

### 7. 性能优化

对于大量数据生成：
- 建议分批生成（每批1000-5000条）
- 确保词典文件已优化（移除重复和无效实体）