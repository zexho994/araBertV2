# 模型评估报告对比工具使用指南

## 功能概述

模型评估报告对比工具允许您合并和对比两个 NER 模型的评估报告，以便直观地查看不同模型在同一地址上的实体识别差异。

## 主要特性

- ✅ 支持 CSV 和 XLSX 格式的输入文件
- ✅ 垂直堆叠格式展示（每个地址占两行，分别显示两个模型的预测）
- ✅ 颜色标记识别差异：
  - 🟢 **绿色**：两个模型预测一致
  - 🟡 **黄色**：两个模型预测不同
  - 🔵 **浅蓝色**：模型名称列
- ✅ 地址之间用空行分隔，便于阅读
- ✅ 自动调整列宽以适应内容

## 使用方法

### 命令行接口

```bash
python ner_cli.py data merge \
  --report-1 <第一个模型报告路径> \
  --report-2 <第二个模型报告路径> \
  --output <输出文件路径> \
  --model1-name <第一个模型名称> \
  --model2-name <第二个模型名称>
```

### 参数说明

| 参数 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `--report-1` | 是 | - | 第一个模型的评估报告路径（支持 .xlsx 或 .csv） |
| `--report-2` | 是 | - | 第二个模型的评估报告路径（支持 .xlsx 或 .csv） |
| `--output` | 是 | - | 输出合并报告的路径（.xlsx 格式） |
| `--model1-name` | 否 | Model-1 | 第一个模型的显示名称 |
| `--model2-name` | 否 | Model-2 | 第二个模型的显示名称 |

### 使用示例

#### 1. 基本使用

```bash
python ner_cli.py data merge \
  --report-1 "data/ner/result/uae/1-1 report.csv" \
  --report-2 "data/ner/result/uae/2-1 report.csv" \
  --output "data/ner/result/uae/comparison_report.xlsx"
```

#### 2. 自定义模型名称

```bash
python ner_cli.py data merge \
  --report-1 "data/ner/result/uae/1-1 report.csv" \
  --report-2 "data/ner/result/uae/2-1 report.csv" \
  --output "data/ner/result/uae/comparison_report.xlsx" \
  --model1-name "Model-v1.0.3" \
  --model2-name "Model-v1.0.6"
```

#### 3. 使用 XLSX 格式输入

```bash
python ner_cli.py data merge \
  --report-1 "data/ner/result/uae/model1_report.xlsx" \
  --report-2 "data/ner/result/uae/model2_report.xlsx" \
  --output "data/ner/result/uae/comparison_report.xlsx" \
  --model1-name "Baseline" \
  --model2-name "Fine-tuned"
```

## Python API 使用

如果您想在 Python 代码中使用该功能：

```python
from src.ner.evaluation.compare_reports import compare_reports

# 对比两个报告
output_path = compare_reports(
    report1_path="data/ner/result/uae/1-1 report.csv",
    report2_path="data/ner/result/uae/2-1 report.csv",
    output_path="data/ner/result/uae/comparison_report.xlsx",
    model1_name="Model-v1.0.3",
    model2_name="Model-v1.0.6"
)

print(f"对比报告已保存到: {output_path}")
```

或者使用类接口：

```python
from src.ner.evaluation.compare_reports import ReportComparator
from src.ner.utils import NERLogger

# 创建 logger
logger = NERLogger()

# 创建对比器
comparator = ReportComparator(logger=logger)

# 生成对比报告
output_path = comparator.generate_comparison_report(
    report1_path="data/ner/result/uae/1-1 report.csv",
    report2_path="data/ner/result/uae/2-1 report.csv",
    output_path="data/ner/result/uae/comparison_report.xlsx",
    model1_name="Model-v1.0.3",
    model2_name="Model-v1.0.6"
)
```

## 输出格式说明

生成的 Excel 文件包含以下内容：

### 数据表格

表格格式（简洁版）：

```
| ADDRESS      | MODEL_NAME | BUILDING | STREET | COMPOUND | SUB_AREA | CITY | EMIRATE | COUNTRY |
|--------------|------------|----------|--------|----------|----------|------|---------|---------|
| 地址1        | Model-1    | 值1      | 值2    | 值3      | ...      | ...  | ...     | ...     |
| (合并单元格) | Model-2    | 值1      | 值2    | 值3      | ...      | ...  | ...     | ...     |
| [空行]       |            |          |        |          |          |      |         |         |
| 地址2        | Model-1    | 值1      | 值2    | 值3      | ...      | ...  | ...     | ...     |
| (合并单元格) | Model-2    | 值1      | 值2    | 值3      | ...      | ...  | ...     | ...     |
```

### 格式特点

- **完整保留原始内容**：所有实体列的内容保持原始格式，包括 "预测: xxx"、"正确: xxx" 等标记
- **ADDRESS 列合并**：每个地址的两行 ADDRESS 单元格会合并为一个，使得表格更加整洁
- **智能颜色标记**：
  - 🟢 绿色：两个模型预测一致（通过提取实际预测值比较）
  - 🟡 黄色：两个模型预测不同（通过提取实际预测值比较）
  - 🔵 浅蓝色：模型名称列
  - 🔷 蓝色：表头
- **垂直居中**：合并的 ADDRESS 单元格内容垂直居中对齐
- **空行分隔**：地址之间用空行分隔，便于阅读
- **简洁布局**：无额外的说明文字，直接展示数据

### 实际示例

假设原始报告包含以下内容：

**报告1 (Model-v1.0.3)**
```
BUILDING: "预测: al yasmin businessman typing centre"
```

**报告2 (Model-v1.0.6)**
```
BUILDING: "预测: businessman typing centre"
```

**合并后的效果**
- 两行都会保留原始格式显示
- 工具识别出两个预测值不同（`al yasmin businessman typing centre` vs `businessman typing centre`）
- 该列用 🟡 **黄色**标记，提示用户这两个模型在此实体的识别结果不同

## 技术实现细节

### 原始内容保留

工具会**完整保留**原始报告中的内容格式，包括 "预测:" 和 "正确:" 等标记，不做任何修改。例如：

- `"预测: al yasmin businessman typing centre"` → 保持原样显示
- `"正确: S128 | 预测: 7 S128"` → 保持原样显示
- `"正确: al yasmin"` → 保持原样显示

### 智能比较逻辑

虽然保留原始格式显示，但在判断两个模型预测是否一致时，工具会智能提取实际预测值进行比较：

- `"预测: value"` → 提取 `"value"` 用于比较
- `"正确: value1 | 预测: value2"` → 提取 `"value2"`（预测值）用于比较
- `"正确: value"` → 提取为空（表示预测为空）

这样既保留了原始信息的完整性，又能准确标记出预测差异。

### 地址匹配

工具按照 ADDRESS 列（通常是第一列）匹配两个报告中的记录。如果某个地址在其中一个报告中不存在，会记录警告并跳过该地址。

### 颜色标记逻辑

对于每个实体类型：
1. 提取两个模型的预测值
2. 比较预测值是否相同（字符串精确匹配）
3. 相同则标记为绿色，不同则标记为黄色

## 故障排除

### 问题：找不到 ADDRESS 列

**原因**：报告格式不符合预期

**解决方案**：确保报告文件的第一列是 ADDRESS 或 Text 列，包含要对比的文本内容。

### 问题：地址匹配不上

**原因**：两个报告中的地址文本不完全一致

**解决方案**：
1. 检查两个报告的 ADDRESS 列内容是否一致
2. 确保使用的是相同的测试数据集

### 问题：无法读取报告文件

**原因**：文件格式不支持或文件损坏

**解决方案**：
1. 确保文件格式为 .xlsx 或 .csv
2. 检查文件是否可以在 Excel 中正常打开
3. 对于 XLSX 文件，确保包含 "=== DETAILED RESULTS ===" 标记

## 相关命令

- [`python ner_cli.py evaluate`](./predict_usage_guide.md) - 生成单个模型的评估报告
- [`python ner_cli.py data convert`](./data_loader_guide.md) - 转换数据格式

## 更新历史

- **v1.2** (2025-10-24): 内容保真版本
  - 🔥 **完整保留原始内容**：保持 "预测: xxx" 和 "正确: xxx" 等原始格式，不做任何修改
  - 🧠 **智能比较逻辑**：在显示原始内容的同时，智能提取实际值进行比较和颜色标记
  - ✨ ADDRESS 列单元格自动合并
  - ✨ 移除顶部颜色图例说明，使布局更简洁
  - ✅ 支持垂直堆叠格式对比
  - ✅ 支持 CSV 和 XLSX 格式输入
  - ✅ 颜色标记预测差异

- **v1.1** (2025-10-24): 优化版本
  - ✨ ADDRESS 列单元格自动合并（每个地址两行合并为一个单元格）
  - ✨ 移除顶部颜色图例说明，使布局更简洁
  - ✅ 支持垂直堆叠格式对比
  - ✅ 支持 CSV 和 XLSX 格式输入
  - ✅ 颜色标记预测差异

- **v1.0** (2025-10-24): 初始版本
  - 支持垂直堆叠格式对比
  - 支持 CSV 和 XLSX 格式输入
  - 颜色标记预测差异

