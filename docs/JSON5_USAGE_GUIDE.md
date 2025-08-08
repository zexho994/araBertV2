# JSON5 支持使用指南

本指南介绍如何在 NER CLI 工具中使用 JSON5 格式的配置模板。

## 功能概述

现在 ConfigManager 支持以下功能：

1. **JSON5 模板支持** - 支持带注释的 JSON5 格式模板文件
2. **向后兼容** - 继续支持传统的 JSON 格式
3. **外部模板** - 支持从任意路径加载模板文件
4. **多格式支持** - 支持 .json、.json5 和 .jsonl 格式

## 安装依赖

确保已安装 json5 库：

```bash
pip install json5>=0.9.6
```

## 使用方法

### 1. 创建 JSON5 模板

在 `data/ner/configs/templates/` 目录下创建 `.json5` 文件：

```json5
// example.json5
{
    // 国家信息
    "country": {
        "code": "example",
        "name": "EXAMPLE"
    },
    
    // 模型配置
    "model": {
        "name": "example-ner-model",
        "type": "bert",
        "pretrained_model": "bert-base-multilingual-cased"
    },
    
    // 训练参数
    "training": {
        "epochs": 5,        // 训练轮数
        "batch_size": 16,   // 批次大小
        "learning_rate": 2e-5  // 学习率
    },
    
    // 数据配置
    "data": {
        "train_file": "train.json",
        "val_file": "val.json",
        "test_file": "test.json"
    },
    
    // 标签配置
    "labels": {
        "num_labels": 9,
        "label_names": [
            "O",           // 非实体
            "B-PER", "I-PER",  // 人名
            "B-ORG", "I-ORG",  // 组织
            "B-LOC", "I-LOC",  // 地点
            "B-MISC", "I-MISC" // 其他
        ],
        "label_mapping": {
            "O": 0,
            "B-PER": 1, "I-PER": 2,
            "B-ORG": 3, "I-ORG": 4,
            "B-LOC": 5, "I-LOC": 6,
            "B-MISC": 7, "I-MISC": 8
        }
    },
    
    // 评估配置
    "evaluation": {
        "metrics": ["precision", "recall", "f1", "accuracy"]
    },
    
    // 输出配置
    "output": {
        "model_dir": "models/example",
        "results_dir": "results/example"
    },
    
    // 硬件配置
    "hardware": {
        "device": "auto",  // auto, cpu, cuda
        "num_workers": 4
    },
    
    // 日志配置
    "logging": {
        "level": "INFO",
        "log_file": "logs/example.log"
    }
}
```

### 2. 使用标准模板创建配置

```bash
# 使用 JSON5 模板
ner config create --country UAE_v1 --template example

# 使用传统 JSON 模板
ner config create --country UAE_v1 --template default
```

### 3. 使用外部模板文件

```bash
# 使用外部 JSON5 模板
ner config create --country UAE_v1 --external-template /path/to/custom_template.json5

# 使用外部 JSON 模板
ner config create --country UAE_v1 --external-template /d:/IdeaProjects/araBertV2/data/dapt/configs/templates/default.json

# 使用外部 JSONL 模板
ner config create --country UAE_v1 --external-template /path/to/template.jsonl
```

### 4. 列出可用模板

```bash
# 列出所有模板（包括 .json 和 .json5）
ner config list --templates
```

### 5. 查看配置

```bash
# 查看创建的配置
ner config show UAE_v1
```

## 编程接口

### 在 Python 代码中使用

```python
from ner.config.manager import ConfigManager

# 初始化配置管理器
config_manager = ConfigManager("data/ner/configs")

# 使用标准模板创建配置
config = config_manager.create_country_config("UAE_v1", "example")

# 使用外部模板创建配置
config = config_manager.create_country_config(
    "UAE_v1", 
    external_template_path="/path/to/custom_template.json5"
)

# 加载外部模板
external_config = config_manager.load_external_template("/path/to/template.json5")

# 列出模板
templates = config_manager.list_templates()
print(f"Available templates: {templates}")
```

## 支持的文件格式

| 格式 | 扩展名 | 注释支持 | 说明 |
|------|--------|----------|------|
| JSON | .json | ❌ | 标准 JSON 格式 |
| JSON5 | .json5 | ✅ | 支持注释和尾随逗号的 JSON5 格式 |
| JSONL | .jsonl | ❌ | JSON Lines 格式（仅读取第一行） |

## 优先级规则

1. **模板加载优先级**：
   - 如果存在 `.json5` 文件且 json5 库可用，优先加载 `.json5`
   - 否则回退到 `.json` 文件

2. **外部模板**：
   - 根据文件扩展名自动检测格式
   - 支持 `.json5`、`.json` 和 `.jsonl` 格式

## 错误处理

- 如果 json5 库未安装，系统会自动回退到标准 JSON 解析
- 不支持的文件格式会抛出明确的错误信息
- JSON 语法错误会提供详细的错误位置信息

## 最佳实践

1. **使用注释**：在 JSON5 模板中添加有意义的注释来解释配置项
2. **保持兼容性**：确保 JSON5 模板在去除注释后仍是有效的 JSON
3. **版本控制**：将模板文件纳入版本控制系统
4. **验证配置**：创建配置后使用 `ner config validate` 验证

## 示例工作流

```bash
# 1. 创建带注释的 JSON5 模板
vim data/ner/configs/templates/uae_custom.json5

# 2. 使用模板创建国家配置
ner config create --country UAE_v1 --template uae_custom

# 3. 验证配置
ner config validate UAE_v1

# 4. 查看配置
ner config show UAE_v1

# 5. 开始训练
ner train --country UAE_v1
```

这样，您就可以充分利用 JSON5 的注释功能来创建更易读和维护的配置模板了！