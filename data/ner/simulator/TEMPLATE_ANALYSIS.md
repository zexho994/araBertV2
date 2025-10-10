# 地址模板分析工具使用指南

## 🎯 工具概述

`analyze_templates.py` 是一个地址模板分析工具，可以：

1. ✅ 分析标注数据中的实体组合模式
2. ✅ 统计各种模板的出现频率
3. ✅ 自动生成符合真实数据分布的模板配置
4. ✅ 提供优化建议

## 🚀 快速使用

### 基础用法

分析标注数据并显示统计结果：

```bash
python3 data/ner/simulator/analyze_templates.py \
    data/ner/raw_data/第三次训练/线上数据-第三次训练-01-标注结果.csv
```

### 生成配置文件

分析数据并生成模板配置：

```bash
python3 data/ner/simulator/analyze_templates.py \
    data/ner/raw_data/第三次训练/线上数据-第三次训练-01-标注结果.csv \
    --output data/ner/simulator/uae/config/templates_from_analysis.json
```

### 自定义参数

```bash
python3 data/ner/simulator/analyze_templates.py <csv_file> \
    --output <output_file>      # 输出配置文件路径
    --top 30                    # 前30个模板
    --min-count 3               # 最小出现次数（过滤低频模板）
```

---

## 📊 输出说明

### 1. 模板统计结果

显示最常见的模板及其出现频率：

```
排名     出现次数       占比         模板
--------------------------------------------------------------------------------
1      35          35.00%    BUILDING + SUB_AREA + CITY + COUNTRY
2      21          21.00%    BUILDING + STREET + SUB_AREA + CITY + COUNTRY
3      12          12.00%    BUILDING + COMPOUND + SUB_AREA + CITY + COUNTRY
```

**说明**：
- 排名1的模板占35%，是最常见的组合
- 前3个模板就覆盖了68%的数据

### 2. 实体出现频率统计

显示各实体类型在数据中的出现率：

```
实体类型            出现次数         出现率        进度条
--------------------------------------------------------------------------------
BUILDING        91            91.00%    █████████████████████████████████████████████
SUB_AREA        98            98.00%    █████████████████████████████████████████████████
CITY            88            88.00%    ████████████████████████████████████████████
COUNTRY         100          100.00%    ██████████████████████████████████████████████████
```

**关键发现**：
- COUNTRY 出现率100% → 必需字段
- SUB_AREA 出现率98% → 几乎必需
- EMIRATE 出现率21% → 可选字段

### 3. 实体组合分析

按实体数量分组显示模板：

```
【4 个实体】共 6 种模板，出现 44 次
  • BUILDING + SUB_AREA + CITY + COUNTRY               (35 次)
  • STREET + SUB_AREA + EMIRATE + COUNTRY              (3 次)

【5 个实体】共 6 种模板，出现 42 次
  • BUILDING + STREET + SUB_AREA + CITY + COUNTRY      (21 次)
  • BUILDING + COMPOUND + SUB_AREA + CITY + COUNTRY    (12 次)
```

**洞察**：
- 4-5个实体的模板最常见
- 帮助确定合理的模板复杂度

### 4. 生成的配置文件

自动生成可直接使用的配置：

```json
{
    "templates": {
        "definitions": [
            {
                "pattern": "{BUILDING} {SUB_AREA} {CITY} {COUNTRY}",
                "weight": 0.373,
                "comment": "出现 35 次"
            },
            {
                "pattern": "{BUILDING} {STREET} {SUB_AREA} {CITY} {COUNTRY}",
                "weight": 0.223,
                "comment": "出现 21 次"
            }
        ]
    }
}
```

**特点**：
- 权重基于真实出现频率
- 权重总和精确为1.0
- 包含出现次数注释

### 5. 配置建议

提供数据驱动的优化建议：

```
1. 数据覆盖度分析:
   • 前10个模板覆盖了 90/100 条记录 (90.0%)
   ✅ 覆盖度很好！使用前10-15个模板即可

2. 必需实体识别（出现率>80%）:
   • BUILDING: 91.0%
   • SUB_AREA: 98.0%
   • CITY: 88.0%
   • COUNTRY: 100.0%

3. 生成数据建议:
   • 真实数据量: 100 条
   • 建议生成: 300 - 500 条
```

---

## 💡 使用场景

### 场景1: 初次配置生成器

**问题**：不知道该使用哪些模板

**解决方案**：
```bash
# 1. 分析所有标注数据
python3 data/ner/simulator/analyze_templates.py \
    data/ner/raw_data/第三次训练/线上数据-第三次训练-标注结果.csv \
    --output data/ner/simulator/uae/config/new_templates.json \

# 2. 查看生成的配置文件
cat data/ner/simulator/uae/config/new_templates.json

# 3. 将templates部分复制到generator_config.json
```

### 场景2: 优化现有配置

**问题**：现有模板配置不符合真实数据分布

**解决方案**：
```bash
# 1. 分析数据，查看真实分布
python3 data/ner/simulator/analyze_templates.py <csv_file>

# 2. 对比工具推荐的权重和现有配置
# 3. 调整generator_config.json中的模板权重
```

### 场景3: 评估数据质量

**问题**：想了解数据的完整性和多样性

**解决方案**：
```bash
# 显示前50个模板
python3 data/ner/simulator/analyze_templates.py <csv_file> --top 50

# 查看实体出现率（识别缺失字段）
# 查看模板分散度（评估多样性）
```

### 场景4: 多文件分析

分析多个标注文件的组合模式：

```bash
# 先合并CSV文件
cat data/ner/raw_data/第三次训练/*.csv > /tmp/all_data.csv

# 然后分析
python3 data/ner/simulator/analyze_templates.py /tmp/all_data.csv --top 30
```

---

## 🎯 实战示例

### 完整工作流程

```bash
# Step 1: 分析真实数据
python3 data/ner/simulator/analyze_templates.py \
    data/ner/raw_data/第三次训练/线上数据-第三次训练-01-标注结果.csv \
    --output data/ner/simulator/uae/config/analyzed_templates.json \

# Step 2: 查看分析结果
# （工具会显示详细的统计信息）

# Step 3: 复制模板配置到主配置文件
# 打开 analyzed_templates.json 和 generator_config.json
# 将 templates.definitions 部分复制过去

# Step 4: 测试生成效果
python3 data/ner/simulator/generator.py

# Step 5: 检查生成的数据
head -20 data/ner/simulator/uae/raw_data/simulated_*.csv
```

### 基于分析结果的调优

根据工具输出的建议调整配置：

```json
{
    "generate_size": 400,  // 基于"建议生成: 300-500条"
    
    "templates": {
        "definitions": [
            // 使用工具生成的模板，权重已优化
            {
                "pattern": "{BUILDING} {SUB_AREA} {CITY} {COUNTRY}",
                "weight": 0.373  // 来自真实数据分析
            }
        ]
    }
}
```

---

## 📈 解读分析结果

### 如何判断数据质量？

**✅ 好的数据特征**：
- 前10个模板覆盖率 > 70%
- 必需字段出现率 > 90%
- 模板种类在10-30之间
- 实体组合有规律（集中在4-6个实体）

**⚠️ 需要注意的情况**：
- 覆盖率 < 50% → 数据太分散
- 模板种类 > 50 → 可能有标注不一致
- 某些必需字段出现率 < 80% → 数据不完整

### 权重分配策略

**工具自动计算的权重**：
- 基于出现频率
- 确保总和为1.0
- 反映真实分布

**手动调整建议**：
```python
# 如果想要更平均的分布
"weight": 0.1  # 不管真实频率，每个模板权重相同

# 如果想强调某些模板
"weight": 0.5  # 手动提高高频模板的权重

# 保持真实分布
"weight": 0.373  # 使用工具计算的权重
```

---

## 🔧 高级用法

### 参数详解

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `csv_file` | 标注数据文件 | - | `data.csv` |
| `--output` | 输出配置文件 | 不保存 | `config.json` |
| `--top` | 显示前N个模板 | 20 | `30` |
| `--min-count` | 最小出现次数 | 1 | `3` |

### 过滤低频模板

只保留出现3次以上的模板：

```bash
python3 data/ner/simulator/analyze_templates.py <csv_file> \
    --min-count 3 \
```

### 查看更多模板

显示前50个模板以全面了解数据：

```bash
python3 data/ner/simulator/analyze_templates.py <csv_file> --top 50
```

---

## 💡 最佳实践

### 1. 定期分析更新

```bash
# 每次收集新数据后重新分析
python3 data/ner/simulator/analyze_templates.py \
    data/ner/raw_data/latest/*.csv \
    --output data/ner/simulator/uae/config/templates_v2.json
```

### 2. 版本管理

```bash
# 为不同版本的模板配置命名
templates_v1.json  # 基于初始100条数据
templates_v2.json  # 基于1000条数据
templates_v3.json  # 优化后的版本
```

### 3. A/B测试

```bash
# 生成两个版本进行对比
# 版本A: 使用真实权重
# 版本B: 使用均匀权重

# 训练模型并比较效果
```

### 4. 数据质量检查

```bash
# 定期运行分析，监控数据变化
python3 data/ner/simulator/analyze_templates.py <csv_file> > analysis_report.txt

# 对比不同时期的报告
diff analysis_report_v1.txt analysis_report_v2.txt
```

---

## 🎓 理解输出

### 覆盖度的含义

```
前10个模板覆盖了 90/100 条记录 (90.0%)
```

**解释**：
- 只需要10个模板就能表示90%的真实数据
- 说明数据模式集中，容易建模
- 生成器使用这10个模板就能很好地模拟真实分布

### 实体出现率的意义

```
COUNTRY: 100.0%  → 必需字段，所有模板都应包含
EMIRATE: 21.0%   → 可选字段，部分模板包含即可
```

**应用**：
- 高出现率实体应该出现在大部分模板中
- 低出现率实体用于增加多样性

### 模板权重的作用

```
"weight": 0.373  // 生成时有37.3%概率选中此模板
"weight": 0.021  // 生成时有2.1%概率选中此模板
```

**影响**：
- 权重越大，该模板生成的数据越多
- 真实权重确保生成数据分布接近真实数据

---

## ❓ 常见问题

**Q: 工具支持哪些CSV格式？**

A: 支持标准的NER标注CSV格式，要求：
- 第一行为表头
- 包含实体类型字段（BUILDING, STREET等）
- UTF-8编码

**Q: 如何处理空值？**

A: 工具自动处理：
- 空字符串、纯空格都视为无效
- 只统计有有效值的实体组合

**Q: 生成的配置可以直接使用吗？**

A: 可以，但建议：
1. 查看并理解配置内容
2. 根据需求微调权重
3. 测试生成效果

**Q: 为什么有些模板只出现1次？**

A: 
- 可能是数据量较小
- 可能是特殊情况
- 使用 `--min-count 2` 过滤掉低频模板

**Q: 权重总和不是1.0怎么办？**

A: 工具会自动归一化，确保总和精确为1.0

---

## 📚 相关文档

- [主README](README.md) - 模拟器总览
- [配置详解](CONFIG_GUIDE.md) - 配置参数说明
- [使用指南](README_USAGE.md) - 生成器使用
- [快速开始](QUICKSTART.md) - 5分钟上手

---

## 🎉 总结

`analyze_templates.py` 让模板配置从"猜测"变为"数据驱动"：

- ✅ **精确**：基于真实数据统计
- ✅ **快速**：几秒钟完成分析
- ✅ **实用**：直接生成可用配置
- ✅ **可靠**：提供优化建议

**立即开始**：

```bash
python3 data/ner/simulator/analyze_templates.py \
    data/ner/raw_data/第三次训练/线上数据-第三次训练-01-标注结果.csv
```

