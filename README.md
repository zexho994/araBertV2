# DAPT - Domain Adaptive Pre-Training for Arabic NLP

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 概述

DAPT (Domain Adaptive Pre-Training) 是一个专门为阿拉伯语自然语言处理设计的领域自适应预训练框架。该项目基于 AraBERT v2 模型，针对不同阿拉伯国家的语言特点和领域需求进行定制化训练，特别优化了命名实体识别 (NER) 任务。

### 主要特性

- 🌍 **多国家支持**: 针对阿联酋、沙特阿拉伯、埃及、摩洛哥等国家的语言特点优化
- 🎯 **领域自适应**: 支持政府、商业、旅游、医疗等多个领域的专门训练
- 🚀 **高性能推理**: 优化的模型推理引擎，支持批量处理和缓存
- 🔧 **易用工具**: 完整的 CLI 工具
- 📊 **全面评估**: 内置评估指标和可视化报告生成
- 🐳 **容器化部署**: 支持 Docker、Kubernetes 等现代化部署方案
- 📈 **监控集成**: 集成 Prometheus、Grafana 等监控工具

### 支持的任务

- **命名实体识别 (NER)**: 识别人名、地名、组织名等实体
- **文本分类**: 文档分类、情感分析等
- **语言模型**: 文本生成、完形填空等

### 支持的国家和方言

| 国家 | 代码 | 方言 | 状态 |
|------|------|------|------|
| 阿联酋 | uae | Gulf Arabic | ✅ 已支持 |

## 快速开始

### 安装

#### 系统要求

- Python 3.8+
- PyTorch 1.12+
- CUDA 11.0+ (可选，用于 GPU 加速)
- 8GB+ RAM (推荐 16GB+)

#### 从源码安装

```bash
# 创建虚拟环境
python -m venv dapt_env
source dapt_env/bin/activate  # Linux/macOS
# 或 dapt_env\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 安装 DAPT
pip install -e .

# 验证安装
dapt --version
```

### 基本使用

#### 1. 训练模型

```bash
# 使用预配置训练 UAE 模型
dapt train --country uae

# 使用自定义配置
dapt train --file configs/countries/uae.yaml

# 指定数据路径和输出目录
dapt train --country uae \
    --data-dir data/uae/ \
    --output-dir ./models/uae_custom
```

#### 2. 评估模型

```bash
# 评估训练好的模型
dapt evaluate --country uae --model-path ./models/uae/best_model
```

#### 3. 进行预测

```bash
# 单个文本预测
dapt predict --country uae \
    --model-path ./models/uae/best_model \
    --text "مرحبا بكم في دولة الإمارات العربية المتحدة"

# 批量预测
dapt predict --country uae \
    --model-path ./models/uae/best_model \
    --input-file texts.txt \
    --output-file predictions.jsonl
```

## 项目结构

```
araBertv2/
├── src/dapt/                   # 核心代码
│   ├── cli/                    # CLI 工具
│   ├── core/                   # 核心训练引擎
│   ├── config/                 # 配置管理
│   ├── data/                   # 数据处理
│   ├── models/                 # 模型管理
│   ├── evaluation/             # 评估模块
│   ├── integration/            # 部署工具
│   └── utils/                  # 工具函数
├── configs/                    # 配置文件
│   ├── countries/              # 国家特定配置
│   └── models/                 # 模型配置
├── data/                       # 训练数据
│   ├── uae/                    # UAE 数据
│   ├── saudi/                  # 沙特数据
│   └── shared/                 # 共享数据
├── models/                     # 训练好的模型
├── docs/                       # 文档
├── examples/                   # 使用示例
├── tests/                      # 测试代码
└── scripts/                    # 脚本工具
```

## 配置说明

### 国家配置文件

每个国家都有专门的配置文件，位于 `configs/countries/` 目录下：

```yaml
# configs/countries/uae.yaml
country:
  code: "uae"
  name: "United Arab Emirates"
  language: "ar"
  dialect: "gulf"

model:
  base_model: "aubmindlab/bert-base-arabertv2"
  task_type: "ner"
  max_length: 512
  special_tokens:
    - "[UAE]"
    - "[DUBAI]"
    - "[ABUDHABI]"

training:
  learning_rate: 2e-5
  batch_size: 16
  epochs: 3
  warmup_steps: 500
  early_stopping_patience: 3

data:
  train_file: "data/uae/train.jsonl"
  validation_file: "data/uae/validation.jsonl"
  test_file: "data/uae/test.jsonl"
  preprocessing:
    normalize_arabic: true
    remove_diacritics: false
    handle_emojis: true

labels:
  ner_labels:
    - "O"
    - "B-PERSON"
    - "I-PERSON"
    - "B-LOCATION"
    - "I-LOCATION"
    - "B-ORGANIZATION"
    - "I-ORGANIZATION"
    - "B-UAE_ENTITY"
    - "I-UAE_ENTITY"

output:
  model_dir: "./models/uae"
  log_dir: "./logs/uae"
  save_steps: 500
  eval_steps: 100
```

### 数据格式

DAPT 支持 JSONL 格式的训练数据：

```jsonl
{"text": "مرحبا بكم في دولة الإمارات العربية المتحدة", "entities": [[13, 41, "COUNTRY"]], "country": "uae"}
{"text": "أهلا وسهلا بكم في دبي", "entities": [[18, 21, "CITY"]], "country": "uae"}
{"text": "الشيخ محمد بن راشد آل مكتوم حاكم دبي", "entities": [[0, 26, "PERSON"], [33, 36, "CITY"]], "country": "uae"}
```

## 高级功能

### 1. 自定义模型训练

```bash
# 使用自定义超参数
dapt train --country uae \
    --learning-rate 3e-5 \
    --batch-size 32 \
    --epochs 5 \
    --warmup-steps 1000

# 从检查点恢复训练
dapt train --country uae \
    --resume-from ./models/uae/checkpoint-1000

# 分布式训练
dapt train --country uae \
    --distributed \
    --nodes 2 \
    --gpus-per-node 4
```

### 2. 模型导出和部署

```bash
# 导出为不同格式
dapt export --country uae \
    --model-path ./models/uae/best_model \
    --format pytorch,onnx,huggingface \
    --output-dir ./exports/uae

# 创建部署包
dapt deploy --country uae \
    --model-path ./models/uae/best_model \
    --deployment-type docker \
    --output-dir ./deployment
```

### 3. 数据处理工具

```bash
# 数据验证
dapt data validate --country uae --data-dir data/uae/

# 数据预处理
dapt data preprocess --country uae \
    --input-dir data/raw/uae/ \
    --output-dir data/processed/uae/

# 数据统计
dapt data stats --country uae --data-dir data/uae/

# 数据格式转换
dapt data convert \
    --input-file data.csv \
    --output-file data.jsonl \
    --input-format csv \
    --output-format jsonl
```

### 4. 超参数搜索

```bash
# 网格搜索
dapt train --country uae \
    --hyperparameter-search grid \
    --search-space configs/search_space.yaml

# 贝叶斯优化
dapt train --country uae \
    --hyperparameter-search bayesian \
    --search-space configs/search_space.yaml \
    --num-trials 50
```

## 监控和日志

### 实时监控

```bash
# 查看实时指标
curl http://localhost:9090/metrics
```

### 日志分析

```bash
# 查看训练日志
tail -f logs/uae/training.log

# 分析错误日志
grep "ERROR" logs/uae/training.log

# 查看性能指标
grep "eval_" logs/uae/training.log | tail -10
```

## 性能基准

### UAE NER 模型性能

| 指标 | 值 |
|------|----|
| F1 Score | 0.923 |
| Precision | 0.918 |
| Recall | 0.928 |
| 推理速度 | ~45ms/文本 |
| 模型大小 | 512MB |

### 系统性能

| 配置 | 吞吐量 (请求/秒) | 延迟 (P95) |
|------|------------------|------------|
| CPU (8 cores) | 15 | 120ms |
| GPU (RTX 3080) | 85 | 45ms |
| GPU (V100) | 120 | 32ms |

## 贡献指南

我们欢迎社区贡献！请遵循以下步骤：

### 开发环境设置

```bash
# 克隆仓库
git clone https://github.com/your-org/araBertv2.git
cd araBertv2

# 安装开发依赖
pip install -r requirements-dev.txt

# 安装 pre-commit hooks
pre-commit install

# 运行测试
pytest tests/
```

### 代码规范

```bash
# 代码格式化
black src/ tests/

# 代码检查
flake8 src/ tests/

# 类型检查
mypy src/

# 导入排序
isort src/ tests/
```



### 添加新国家支持

1. 创建国家配置文件：`configs/countries/new_country.yaml`
2. 准备训练数据：`data/new_country/`
3. 添加测试用例：`tests/test_new_country.py`
4. 更新文档：`docs/`

## 常见问题

### Q: 如何添加自定义实体类型？

A: 在国家配置文件中的 `labels.ner_labels` 部分添加新的标签：

```yaml
labels:
  ner_labels:
    - "O"
    - "B-CUSTOM_ENTITY"
    - "I-CUSTOM_ENTITY"
```

### Q: 如何处理内存不足错误？

A: 尝试以下解决方案：

```bash
# 减少批大小
dapt train --country uae --batch-size 8

# 启用梯度累积
dapt train --country uae --batch-size 8 --gradient-accumulation-steps 2

# 使用混合精度训练
dapt train --country uae --fp16
```

### Q: 如何提高推理速度？

A: 考虑以下优化方案：

1. 使用 GPU 推理
2. 启用模型缓存
3. 使用批量预测
4. 模型量化
5. 使用 TensorRT 优化

### Q: 支持哪些阿拉伯语方言？

A: 目前主要支持：
- 海湾阿拉伯语 (UAE, Kuwait, Qatar)

## 参考

- [AraBERT](https://github.com/aub-mind/arabert) - 基础模型
- [Hugging Face Transformers](https://github.com/huggingface/transformers) - 模型框架
- [PyTorch](https://pytorch.org/) - 深度学习框架

# NER CLI - Named Entity Recognition Command Line Tool

一个功能强大的命名实体识别（NER）命令行工具，专为阿拉伯语和多语言文本处理而设计，特别针对UAE地址解析进行了优化。

## 🌟 特性

- **多语言支持**: 支持阿拉伯语、英语等多种语言
- **UAE地址解析**: 专门针对阿联酋地址格式优化，支持23个地址实体标签
- **灵活配置**: 基于JSON的配置系统，支持国家特定配置
- **完整CLI**: 提供训练、评估、预测、配置管理等完整功能
- **BERT集成**: 基于Transformers库，支持多种预训练模型
- **数据处理**: 支持CoNLL、JSON、CSV等多种数据格式
- **模型管理**: 完整的模型版本管理和备份功能
- **评估指标**: 详细的评估报告和性能分析

## 📦 安装

### 从源码安装

```bash
# 克隆项目
git clone <repository-url>
cd araBertv2

# 安装依赖
pip install -r requirements.txt

# 安装项目
pip install -e .
```

### 使用pip安装

```bash
pip install ner
```

## 🚀 快速开始

### 1. 检查系统状态

```bash
ner status
```

### 2. 查看可用配置

```bash
ner config list
```

### 3. 训练UAE地址解析模型

```bash
# 准备训练数据（JSON格式）
ner train --country uae --data-path ./data/uae_train.json --output-dir ./models/uae_model
```

### 4. 评估模型

```bash
ner evaluate --model-path ./models/uae_model --data-path ./data/uae_test.json
```

### 5. 进行预测

```bash
# 单个文本预测
ner predict --model-path ./models/uae_model --text "123 Sheikh Zayed Road, Dubai Marina, Dubai"

# 文件批量预测
ner predict --model-path ./models/uae_model --file ./data/test_texts.txt
```

## 📊 UAE地址实体标签

本工具支持以下23个UAE地址相关的实体标签：

| 标签 | 描述 | 示例 |
|------|------|------|
| COUNTRY | 国家 | United Arab Emirates, UAE |
| EMIRATE | 酋长国 | Dubai, Abu Dhabi, Sharjah |
| CITY | 城市 | Dubai, Abu Dhabi |
| AREA | 区域 | Dubai Marina, Downtown |
| DISTRICT | 地区 | Business Bay, DIFC |
| NEIGHBORHOOD | 社区 | JBR, Palm Jumeirah |
| STREET | 街道 | Sheikh Zayed Road |
| BUILDING | 建筑物 | Burj Khalifa, Emirates Towers |
| BUILDING_NUMBER | 建筑编号 | Tower 1, Block A |
| FLOOR | 楼层 | 15th Floor, Ground Floor |
| APARTMENT | 公寓号 | Apt 1502, Unit 25 |
| VILLA_NUMBER | 别墅号 | Villa 123, House 45 |
| PLOT_NUMBER | 地块号 | Plot 456, Land 789 |
| MAKANI_NUMBER | Makani号码 | 1234567890+ |
| PO_BOX | 邮政信箱 | P.O. Box 12345 |
| POSTAL_CODE | 邮政编码 | 12345, DXB001 |
| LANDMARK | 地标 | Near Mall of Emirates |
| DIRECTION | 方向 | North, South, Opposite |
| ROAD_TYPE | 道路类型 | Road, Street, Avenue |
| LOCATION_TYPE | 位置类型 | Office, Residence |
| COORDINATE | 坐标 | 25.2048, 55.2708 |
| PHONE | 电话号码 | +971-4-1234567 |
| WEBSITE | 网站 | www.example.ae |

## 🛠️ 命令详解

### 训练命令

```bash
ner train [OPTIONS]

选项:
  --country, -c TEXT        国家配置 (必需)
  --data-path, -d TEXT      训练数据路径 (必需)
  --val-data-path TEXT      验证数据路径
  --output-dir, -o TEXT     输出目录
  --epochs INTEGER          训练轮数
  --batch-size INTEGER      批次大小
  --learning-rate FLOAT     学习率
  --resume TEXT             从检查点恢复训练
```

### 评估命令

```bash
ner evaluate [OPTIONS]

选项:
  --model-path, -m TEXT     模型路径 (必需)
  --data-path, -d TEXT      评估数据路径 (必需)
  --output-dir, -o TEXT     输出目录
  --batch-size INTEGER      批次大小
```

### 预测命令

```bash
ner predict [OPTIONS]

选项:
  --model-path, -m TEXT           模型路径 (必需)
  --text, -t TEXT                 要分析的文本
  --file, -f TEXT                 包含文本的文件
  --output-format [json|text|conll]  输出格式 (默认: json)
  --confidence-threshold FLOAT    置信度阈值 (默认: 0.5)
```

### 配置管理

```bash
# 列出所有配置
ner config list

# 显示特定配置
ner config show uae

# 创建新配置
ner config create --country egypt --template address_ner

# 验证配置
ner config validate uae
```

### 数据处理

```bash
# 验证数据格式
ner data validate --data-path ./data/train.json

# 转换数据格式
ner data convert --input-path ./data/train.json --output-path ./data/train.conll --output-format conll

# 分割数据集
ner data split --data-path ./data/full_data.json --output-dir ./data/splits --train-ratio 0.8 --val-ratio 0.1
```

### 模型管理

```bash
# 列出所有模型
ner model list

# 显示模型信息
ner model info uae_model_v1

# 删除模型
ner model delete old_model --force
```

## 📁 项目结构

```
araBertv2/
├── ner_cli.py                 # 主入口文件
├── requirements.txt           # 依赖文件
├── setup.py                  # 安装脚本
├── README.md                 # 项目文档
├── data/ner/                 # 数据目录
│   ├── configs/              # 配置文件
│   │   ├── templates/        # 配置模板
│   │   └── countries/        # 国家特定配置
│   ├── datasets/             # 数据集
│   ├── models/               # 训练好的模型
│   ├── outputs/              # 输出结果
│   ├── logs/                 # 日志文件
│   └── checkpoints/          # 训练检查点
└── src/                      # 源代码
    └── ner/                  # NER模块
        ├── __init__.py
        ├── cli/              # CLI命令
        ├── config/           # 配置管理
        ├── data/             # 数据处理
        ├── training/         # 训练引擎
        ├── models/           # 模型管理
        ├── evaluation/       # 评估模块
        └── utils/            # 工具函数
```

## 📋 数据格式

### JSON格式

```json
[
  {
    "text": "123 Sheikh Zayed Road, Dubai Marina, Dubai",
    "entities": [
      {"start": 0, "end": 3, "label": "BUILDING_NUMBER", "text": "123"},
      {"start": 4, "end": 21, "label": "STREET", "text": "Sheikh Zayed Road"},
      {"start": 23, "end": 35, "label": "AREA", "text": "Dubai Marina"},
      {"start": 37, "end": 42, "label": "CITY", "text": "Dubai"}
    ]
  }
]
```

### CoNLL格式

```
123 B-BUILDING_NUMBER
Sheikh B-STREET
Zayed I-STREET
Road I-STREET
, O
Dubai B-AREA
Marina I-AREA
, O
Dubai B-CITY

```

## ⚙️ 配置文件

配置文件使用JSON格式，包含以下主要部分：

- **country**: 国家信息
- **model**: 模型配置（预训练模型、架构等）
- **training**: 训练参数
- **data**: 数据处理配置
- **labels**: 实体标签定义
- **evaluation**: 评估指标
- **output**: 输出设置
- **hardware**: 硬件配置
- **logging**: 日志配置

## 🔧 高级用法

### 自定义配置

1. 复制现有模板：
```bash
cp data/ner/configs/templates/address_ner.json data/ner/configs/countries/my_country.json
```

2. 修改配置文件以适应您的需求

3. 验证配置：
```bash
ner config validate my_country
```

### 批量处理

```bash
# 批量预测多个文件
for file in data/*.txt; do
    ner predict --model-path ./models/uae_model --file "$file" --output-format json > "results/$(basename "$file" .txt).json"
done
```

### 模型微调

```bash
# 在现有模型基础上继续训练
ner train --country uae --data-path ./data/additional_data.json --resume ./models/uae_model/checkpoint-1000
```

## 📈 性能优化

### GPU加速

```bash
# 安装GPU支持
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 使用GPU训练
ner train --country uae --data-path ./data/train.json --batch-size 32
```

### 内存优化

- 减少批次大小：`--batch-size 8`
- 使用梯度累积：在配置文件中设置 `gradient_accumulation_steps`
- 启用混合精度训练：在配置文件中设置 `fp16: true`

## 🐛 故障排除

### 常见问题

1. **导入错误**：确保已正确安装所有依赖
2. **内存不足**：减少批次大小或使用更小的模型
3. **CUDA错误**：检查GPU驱动和CUDA版本兼容性
4. **数据格式错误**：使用 `ner data validate` 检查数据格式

### 日志分析

```bash
# 查看详细日志
ner --verbose --log-level DEBUG train --country uae --data-path ./data/train.json

# 查看日志文件
tail -f logs/ner_cli.log
```


## 🙏 致谢

- [Transformers](https://huggingface.co/transformers/) - 预训练模型支持
- [AraBERT](https://github.com/aub-mind/arabert) - 阿拉伯语BERT模型
- [seqeval](https://github.com/chakki-works/seqeval) - NER评估指标