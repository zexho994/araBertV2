# DAPT 极小数量训练使用指南

## 📖 简介

本指南专门为第一次接触 DAPT (Domain Adaptive Pre-Training) 项目的用户设计，提供最简单、最直接的极小数量训练方法。通过本指南，您可以快速上手并完成一个基础的 DAPT 领域自适应预训练任务。

**重要说明**: DAPT 是无监督的领域自适应预训练，不需要标注数据，只需要纯文本数据。

## 🎯 什么是极小数量训练？

极小数量训练是指使用最少的数据样本（通常 50-200 条）进行快速的模型训练和验证，主要用于：
- 快速验证训练流程
- 测试配置是否正确
- 学习和理解 DAPT 训练过程
- 原型开发和概念验证

## 📁 必需文件和目录结构

### 1. 核心目录结构

```
araBertv2/
├── data/dapt/                   # DAPT 工作目录
│   ├── training_data/uae/      # UAE 训练数据目录
│   │   ├── train.txt           # 训练数据（必需）
│   │   ├── validation.txt      # 验证数据（必需）
│   │   └── test.txt            # 测试数据（必需）
│   ├── models/                 # 模型输出目录
│   ├── logs/                   # 日志目录
│   └── configs/                # 配置文件目录
└── dapt_cli.py                 # CLI 工具（已存在）
```

### 2. 必需的数据文件

您需要准备以下三个纯文本格式的数据文件：

#### 训练数据

 `data/dapt/training_data/uae/train.txt`
```
شارع الشيخ زايد، دبي، الإمارات العربية المتحدة
مركز دبي التجاري العالمي، دبي
جامعة الإمارات العربية المتحدة، العين
برج خليفة، وسط مدينة دبي
مطار دبي الدولي، دبي
```

#### 验证数据
`data/dapt/training_data/uae/validation.txt` 
```
شارع الكورنيش، أبوظبي
مول الإمارات، دبي
جزيرة ياس، أبوظبي
```

#### 测试数据
`data/dapt/training_data/uae/test.txt`
```
مدينة دبي الطبية، دبي
منطقة دبي مارينا، دبي
قصر الإمارات، أبوظبي
```

## 🚀 最简单的执行步骤

### 步骤 1: 环境准备

```bash
# 1. 进入项目目录
cd /Users/zexho/Documents/python_script/araBertv2

# 2. 启动虚拟环境
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# 3. 安装依赖（如果还未安装）
pip install -r requirements.txt

# 4. 安装 dapt_cli
pip install -e .

# 5. 验证 CLI 工具
dapt --help
```

### 步骤 2: 准备极小数据集

```bash
# 1. 确保数据目录存在
mkdir -p data/dapt/training_data/uae

# 2. 创建最小训练数据集（纯文本格式）
# 您可以直接复制上面提供的示例数据到对应的 .txt 文件中
```

### 步骤 3: 验证配置

```bash
# 验证 UAE 配置文件是否正确
dapt config validate --country uae
```

### 步骤 4: 数据验证

```bash
# 验证数据格式是否正确
dapt data validate --country uae --input-file data/dapt/training_data/uae/train.txt
```

### 步骤 5: 开始极小数量训练

```bash
# 使用最小配置进行 DAPT 预训练（仅 2 个 epoch，适合快速测试）
dapt train --country uae --epochs 2 --batch-size 2
```

### 步骤 6: 查看训练状态

```bash
# 查看训练日志和状态
dapt status --country uae
```

### 步骤 7: 评估模型

```bash
# 评估模型性能
dapt evaluate --country uae --model data/dapt/models/uae/final_model
```

## 📊 数据格式说明

### 纯文本格式要求

- **文件格式**: `.txt` 文件
- **编码**: UTF-8
- **内容**: 每行一条阿拉伯语文本
- **语言**: 主要为阿拉伯语，可包含少量英语
- **领域**: UAE 相关的地址、地名、机构名称等

### 数据示例

```
شارع الشيخ زايد، دبي، الإمارات العربية المتحدة
مركز دبي التجاري العالمي
جامعة الإمارات العربية المتحدة، العين
برج خليفة، وسط مدينة دبي
مطار دبي الدولي
شارع الكورنيش، أبوظبي
مول الإمارات، دبي
جزيرة ياس، أبوظبي
```

## ⚠️ 常见问题和解决方案

### 问题 1: 数据格式错误
**错误信息**: `Invalid data format`
**解决方案**: 
- 确保文件是 UTF-8 编码的纯文本文件
- 检查文件扩展名是否为 `.txt`
- 确保每行包含有效的阿拉伯语文本

### 问题 2: 内存不足
**错误信息**: `CUDA out of memory`
**解决方案**:
- 减小 batch_size（如 `--batch-size 1`）
- 使用 CPU 训练（添加 `--device cpu`）

### 问题 3: 配置文件找不到
**错误信息**: `Config file not found`
**解决方案**:
- 确保 `data/dapt/configs/countries/uae.json` 文件存在
- 检查文件路径是否正确

## 🎯 训练参数调优建议

### 极小数据集推荐参数

```bash
# 最小配置（快速测试）
dapt train --country uae \
    --epochs 2 \
    --batch-size 1 \
    --learning-rate 5e-5 \
    --warmup-steps 10

# 稍大配置（更好效果）
dapt train --country uae \
    --epochs 5 \
    --batch-size 2 \
    --learning-rate 2e-5 \
    --warmup-steps 50
```

### 参数说明

- `--epochs`: 训练轮数（极小数据建议 2-5）
- `--batch-size`: 批大小（极小数据建议 1-4）
- `--learning-rate`: 学习率（建议 2e-5 到 5e-5）
- `--warmup-steps`: 预热步数（建议总步数的 10%）

## 📈 预期结果

### 训练输出示例

```
🇦🇪 Starting UAE DAPT Training...
📊 Training Data: 100 samples
📊 Validation Data: 30 samples
📊 Test Data: 30 samples

Epoch 1/2:
  Training Loss: 2.456
  Validation Loss: 2.123
  Perplexity: 11.67

Epoch 2/2:
  Training Loss: 1.987
  Validation Loss: 1.876
  Perplexity: 6.52

✅ Training completed!
📊 Final Perplexity: 6.52
💾 Model saved to: data/dapt/models/uae/
```

## 🔄 下一步建议

完成极小数量训练后，您可以：

1. **扩展数据集**: 增加更多训练样本（建议 1000-10000 条）
2. **调优参数**: 尝试不同的学习率和批大小
3. **领域适配**: 针对特定领域（商业、政府、旅游）收集数据
4. **模型评估**: 使用更大的测试集进行全面评估
5. **下游任务**: 将预训练好的模型用于具体的 NLP 任务

## 📚 相关文档

- [完整训练指南](../docs/quick_start.md)
- [CLI 工具详细说明](../docs/CLI_USAGE.md)
- [配置文件说明](../data/dapt/configs/countries/uae.json)
- [示例代码](../examples/uae_training_example.py)

## 💡 提示

- DAPT 是无监督预训练，专注于学习领域特定的语言表示
- 极小数量训练主要用于验证流程，实际应用需要更多数据
- 建议先在 CPU 上测试，确认流程无误后再使用 GPU
- 保存好训练日志，便于问题排查和参数调优
- 定期备份训练好的模型和配置文件
- 预训练完成后，可以将模型用于下游的 NER、分类等任务

---

**祝您训练顺利！如有问题，请查看项目文档或提交 Issue。**