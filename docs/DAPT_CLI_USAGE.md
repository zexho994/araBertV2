# DAPT CLI 工具使用指南

## 概述

DAPT (Domain Adaptive Pre-Training) CLI 工具是一个强大的命令行界面，用于训练、评估和部署针对特定国家和领域的阿拉伯语 NLP 模型。

## 安装

### 从源码安装

```bash
# 安装依赖
pip install -r requirements.txt

# 安装 DAPT 包
pip install -e .
```

### 验证安装

```bash
dapt --version
dapt --help
```

## 基本用法

### 命令结构

```bash
dapt <command> [subcommand] [options]
```

### 主要命令

- `train` - 训练模型
- `evaluate` - 评估模型
- `predict` - 进行预测
- `export` - 导出模型

- `deploy` - 部署模型
- `config` - 配置管理
- `data` - 数据处理

## 详细命令说明

### 1. 训练命令 (train)

#### 基本训练

```bash
# 使用默认配置训练 UAE 模型
dapt train --country uae

# 指定配置文件
dapt train --file data/dapt/configs/countries/uae.yaml

# 指定数据路径
dapt train --country uae --data-dir data/uae/

# 自定义输出目录
dapt train --country uae --output-dir ./models/uae_custom
```

#### 高级训练选项

```bash
# 从检查点恢复训练
dapt train --country uae --resume-from ./models/uae/checkpoint-1000

# 指定 GPU
dapt train --country uae --gpu 0,1

# 分布式训练
dapt train --country uae --distributed --nodes 2 --gpus-per-node 4

# 调试模式
dapt train --country uae --debug --max-steps 100
```

#### 训练参数覆盖

```bash
# 覆盖学习率
dapt train --country uae --learning-rate 2e-5

# 覆盖批大小
dapt train --country uae --batch-size 16

# 覆盖训练轮数
dapt train --country uae --epochs 5

# 多个参数覆盖
dapt train --country uae \
    --learning-rate 2e-5 \
    --batch-size 16 \
    --epochs 5 \
    --warmup-steps 1000
```

### 2. 评估命令 (evaluate)

#### 基本评估

```bash
# 评估训练好的模型
dapt evaluate --country uae --model-path ./data/dapt/models/uae/best_model

# 指定测试数据
dapt evaluate --country uae --model-path ./data/dapt/models/uae/best_model --test-data data/uae/test.jsonl

# 评估多个数据集
dapt evaluate --country uae --model-path ./data/dapt/models/uae/best_model \
    --test-data data/dapt/training_data/uae/test_business.jsonl,data/uae/test_government.jsonl
```

#### 评估选项

```bash
# 生成详细报告
dapt evaluate --country uae --model-path ./models/uae/best_model --detailed-report

# 指定输出格式
dapt evaluate --country uae --model-path ./models/uae/best_model \
    --output-format json,html,csv

# 保存预测结果
dapt evaluate --country uae --model-path ./models/uae/best_model \
    --save-predictions ./results/uae_predictions.jsonl
```

### 3. 预测命令 (predict)

#### 文本预测

```bash
# 单个文本预测
dapt predict --country uae --model-path ./models/uae/best_model \
    --text "مرحبا بكم في دولة الإمارات العربية المتحدة"

# 从文件预测
dapt predict --country uae --model-path ./models/uae/best_model \
    --input-file input.txt --output-file predictions.jsonl

# 批量预测
dapt predict --country uae --model-path ./models/uae/best_model \
    --input-file batch_input.jsonl --batch-size 32
```

#### 预测选项

```bash
# 指定输出格式
dapt predict --country uae --model-path ./models/uae/best_model \
    --text "النص العربي" --output-format json

# 包含置信度分数
dapt predict --country uae --model-path ./models/uae/best_model \
    --text "النص العربي" --include-scores

# 设置置信度阈值
dapt predict --country uae --model-path ./models/uae/best_model \
    --text "النص العربي" --confidence-threshold 0.8
```

### 4. 导出命令 (export)

#### 模型导出

```bash
# 导出为 PyTorch 格式
dapt export --country uae --model-path ./models/uae/best_model \
    --format pytorch --output-dir ./exports/uae_pytorch

# 导出为 ONNX 格式
dapt export --country uae --model-path ./models/uae/best_model \
    --format onnx --output-dir ./exports/uae_onnx

# 导出为 Hugging Face 格式
dapt export --country uae --model-path ./models/uae/best_model \
    --format huggingface --output-dir ./exports/uae_hf
```

#### 导出选项

```bash
# 模型优化
dapt export --country uae --model-path ./models/uae/best_model \
    --format onnx --optimize --output-dir ./exports/uae_optimized

# 模型量化
dapt export --country uae --model-path ./models/uae/best_model \
    --format pytorch --quantize --output-dir ./exports/uae_quantized

# 包含部署配置
dapt export --country uae --model-path ./models/uae/best_model \
    --format pytorch --include-deployment-config --output-dir ./exports/uae_deploy
```



### 6. 部署命令 (deploy)

#### Docker 部署

```bash
# 创建 Docker 部署包
dapt deploy --country uae --model-path ./models/uae/best_model \
    --deployment-type docker --output-dir ./deployment

# 构建并运行 Docker 容器
dapt deploy --country uae --model-path ./models/uae/best_model \
    --deployment-type docker --build-image --run-container
```

#### 云部署

```bash
# AWS 部署配置
dapt deploy --country uae --model-path ./models/uae/best_model \
    --deployment-type aws --target-env prod

# GCP 部署配置
dapt deploy --country uae --model-path ./models/uae/best_model \
    --deployment-type gcp --target-env staging

# Azure 部署配置
dapt deploy --country uae --model-path ./models/uae/best_model \
    --deployment-type azure --target-env dev
```

### 7. 配置命令 (config)

#### 配置管理

```bash
# 列出所有配置
dapt config list

# 显示特定国家配置
dapt config show --country uae

# 验证配置文件
dapt config validate --file data/dapt/configs/countries/uae.yaml

# 创建新配置模板
dapt config create --country saudi --template uae
```

#### 配置编辑

```bash
# 设置配置值
dapt config set --country uae --key training.learning_rate --value 2e-5

# 获取配置值
dapt config get --country uae --key training.learning_rate

# 重置配置为默认值
dapt config reset --country uae
```

### 8. 数据命令 (data)

#### 数据处理

```bash
# 验证数据格式
dapt data validate --country uae --data-dir data/dapt/training_data/uae/

# 数据预处理
dapt data preprocess --country uae --input-dir data/dapt/training_data/uae/raw/ \
    --output-dir data/dapt/training_data/uae/processed/

# 数据统计
dapt data stats --country uae --data-dir data/dapt/training_data/uae/
```

#### 数据转换

```bash
# 格式转换
dapt data convert --input-file data.csv --output-file data.jsonl \
    --input-format csv --output-format jsonl

# 数据分割
dapt data split --input-file data.jsonl --train-ratio 0.8 \
    --val-ratio 0.1 --test-ratio 0.1

# 数据增强
dapt data augment --country uae --input-file data/uae/train.jsonl \
    --output-file data/uae/train_augmented.jsonl --augment-ratio 0.2
```

## 配置文件

### 国家配置文件结构

```yaml
# data/dapt/configs/countries/uae.yaml
country:
  code: "uae"
  name: "United Arab Emirates"
  language: "ar"
  dialect: "gulf"

model:
  base_model: "aubmindlab/bert-base-arabertv2"
  task_type: "ner"
  max_length: 512
  
training:
  learning_rate: 2e-5
  batch_size: 16
  epochs: 3
  warmup_steps: 500
  
data:
  train_file: "data/dapt/training_data/uae/train.jsonl"
  validation_file: "data/dapt/training_data/uae/validation.jsonl"
  test_file: "data/dapt/training_data/uae/test.jsonl"
  
output:
  model_dir: "./models/uae"
  log_dir: "./logs/uae"
```



## 环境变量

### 常用环境变量

```bash
# 设置默认配置目录
export DAPT_CONFIG_DIR="/path/to/configs"

# 设置默认数据目录
export DAPT_DATA_DIR="/path/to/data"

# 设置默认输出目录
export DAPT_OUTPUT_DIR="/path/to/output"

# 设置日志级别
export DAPT_LOG_LEVEL="INFO"

# 设置 GPU 设备
export CUDA_VISIBLE_DEVICES="0,1"

# 设置 Hugging Face 缓存目录
export HF_HOME="/path/to/hf_cache"
```

## 使用示例

### 完整训练流程

```bash
# 1. 验证数据
dapt data validate --country uae --data-dir data/uae/

# 2. 开始训练
dapt train --country uae --file data/dapt/configs/countries/uae.yaml

# 3. 评估模型
dapt evaluate --country uae --model ./data/dapt/models/uae/final_model --eval-dir ./data/dapt/evaluation/uae/report

# 4. 导出模型
dapt export --country uae --model ./data/dapt/models/uae/final_model \
    --format pytorch --output-dir ./exports/uae
```

## 故障排除

### 常见问题

#### 1. 内存不足错误

```bash
# 减少批大小
dapt train --country uae --batch-size 8

# 启用梯度累积
dapt train --country uae --batch-size 8 --gradient-accumulation-steps 2

# 使用混合精度训练
dapt train --country uae --fp16
```

#### 2. CUDA 错误

```bash
# 检查 GPU 可用性
nvidia-smi

# 指定特定 GPU
dapt train --country uae --gpu 0

# 使用 CPU 训练
dapt train --country uae --device cpu
```

#### 3. 数据格式错误

```bash
# 验证数据格式
dapt data validate --country uae --data-dir data/uae/

# 查看数据统计
dapt data stats --country uae --data-dir data/uae/

# 转换数据格式
dapt data convert --input-file data.csv --output-file data.jsonl \
    --input-format csv --output-format jsonl
```

#### 4. 配置文件错误

```bash
# 验证配置文件
dapt config validate --file data/dapt/configs/countries/uae.yaml

# 显示配置内容
dapt config show --country uae

```
## 最佳实践

### 1. 数据准备

- 确保数据格式正确且一致
- 进行数据质量检查和清洗
- 合理分割训练、验证和测试集
- 考虑数据增强技术

### 2. 模型训练

- 从小规模实验开始
- 监控训练过程和指标
- 使用早停防止过拟合
- 保存多个检查点

### 3. 模型评估

- 在多个测试集上评估
- 分析错误案例
- 比较不同模型版本
- 考虑领域特定指标

### 4. 部署和监控

- 使用容器化部署
- 设置健康检查
- 监控性能指标
- 准备回滚策略

### 5. 版本管理

- 记录模型版本和配置
- 保存训练日志和指标
- 使用 Git 管理代码版本
- 建立模型注册表

## 支持和贡献

### 获取帮助

- 查看内置帮助：`dapt --help`
- 查看命令帮助：`dapt train --help`
- 查看配置示例：`dapt config show --country uae`

### 报告问题

如果遇到问题，请提供以下信息：

1. 完整的命令和参数
2. 错误消息和堆栈跟踪
3. 系统环境信息
4. 配置文件内容
5. 数据样本（如果可能）

### 贡献代码

欢迎贡献代码和改进建议：

1. Fork 项目仓库
2. 创建功能分支
3. 提交更改
4. 创建 Pull Request
5. 等待代码审查

---

更多详细信息请参考项目文档和源代码注释。