# AraBERTv2 快速入门指南

## 🚀 AraBERTv2 模型简介

AraBERTv2是专门为阿拉伯语设计的BERT模型，由AUB MIND Lab开发。本项目使用AraBERTv2进行阿拉伯语地址解析。

## 📋 模型版本

| 模型 | 参数量 | 用途 |
|------|--------|------|
| `aubmindlab/bert-base-arabertv2` | 110M | 基础版本，速度快 |
| `aubmindlab/bert-large-arabertv2` | 340M | 大型版本，精度高 |

## 🛠️ 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖：
- `torch>=1.9.0`
- `transformers>=4.20.0`
- `arabic-reshaper>=2.1.3`
- `python-bidi>=0.4.2`

## 💡 基础使用

### 1. 直接使用AraBERTv2

```python
from transformers import AutoTokenizer, AutoModel
import torch

# 加载模型和分词器
model_name = "aubmindlab/bert-base-arabertv2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 处理阿拉伯语文本
text = "شارع الملك فهد، حي الملز، الرياض"
inputs = tokenizer(text, return_tensors="pt")

# 获取编码
with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state
```

### 2. 使用项目中的NER模型

```python
from src.model.arabertv2_ner import AraBERTv2NER, AraBERTv2Tokenizer
from configs.config import MODEL_CONFIG

# 创建NER模型
model = AraBERTv2NER(
    model_name=MODEL_CONFIG["model_name"],
    num_labels=MODEL_CONFIG["num_labels"]
)

# 创建分词器
tokenizer = AraBERTv2Tokenizer(MODEL_CONFIG["model_name"])
```

## 🎯 地址解析任务

### 支持的实体类型

| 实体类型 | 描述 | 示例 |
|----------|------|------|
| STREET | 街道名称 | شارع الملك فهد |
| BUILDING | 建筑物/门牌号 | مبنى رقم 123 |
| DISTRICT | 地区/区域 | حي الملز |
| CITY | 城市 | الرياض |
| COUNTRY | 国家 | المملكة العربية السعودية |
| POSTAL_CODE | 邮政编码 | 21411 |

### BIO标注格式

- `B-ENTITY`: 实体开始
- `I-ENTITY`: 实体内部
- `O`: 非实体

示例：
```
شارع    -> B-STREET
الملك    -> I-STREET
فهد     -> I-STREET
،       -> O
حي      -> B-DISTRICT
الملز    -> I-DISTRICT
```

## 🔄 完整训练流程

### 1. 数据预处理
```bash
python src/data_processing/preprocess.py
```

### 2. 模型训练
```bash
python src/training/train.py
```

### 3. 模型评估
```bash
python src/training/evaluate.py
```

### 4. 运行演示
```bash
python run_demo.py
```

## 📊 数据格式

### 输入数据格式
```json
{
    "text": "شارع الملك فهد، حي الملز، الرياض",
    "entities": [
        {
            "start": 0,
            "end": 13,
            "label": "STREET",
            "text": "شارع الملك فهد"
        },
        {
            "start": 16,
            "end": 24,
            "label": "DISTRICT", 
            "text": "حي الملز"
        },
        {
            "start": 27,
            "end": 34,
            "label": "CITY",
            "text": "الرياض"
        }
    ]
}
```

### 处理后的BIO格式
```json
{
    "tokens": ["شارع", "الملك", "فهد", "،", "حي", "الملز", "،", "الرياض"],
    "labels": ["B-STREET", "I-STREET", "I-STREET", "O", "B-DISTRICT", "I-DISTRICT", "O", "B-CITY"],
    "label_ids": [1, 2, 2, 0, 5, 6, 0, 7]
}
```

## ⚙️ 配置说明

### 模型配置 (configs/config.py)
```python
MODEL_CONFIG = {
    "model_name": "aubmindlab/bert-base-arabertv2",
    "model_name_large": "aubmindlab/bert-large-arabertv2", 
    "max_length": 512,
    "num_labels": 13
}
```

### 训练配置
```python
TRAINING_CONFIG = {
    "batch_size": 16,
    "learning_rate": 2e-5,
    "num_epochs": 10,
    "warmup_steps": 500
}
```

## 🚀 性能优化

### 硬件要求
- **GPU内存**: 4GB+ (Base), 8GB+ (Large)
- **系统内存**: 8GB+
- **存储空间**: 2GB+

### 优化建议
1. **模型选择**: Base模型适合快速原型，Large模型适合生产环境
2. **批处理大小**: 根据GPU内存调整batch_size
3. **混合精度**: 使用fp16减少内存使用
4. **梯度累积**: 处理大批次数据

## 📝 示例代码

### 快速预测示例
```python
# 加载训练好的模型
from src.training.evaluate import ModelEvaluator

# 假设已有训练好的模型
model_path = "outputs/models/best_model_epoch_1"
evaluator = ModelEvaluator(model_path)

# 预测新文本
text = "شارع الأمير محمد، الدمام، السعودية"
tokens, labels = evaluator.predict_text(text)

# 提取实体
entities = evaluator.extract_entities(tokens, labels)
for entity in entities:
    print(f"{entity['text']} ({entity['label']})")
```

## 🔧 故障排除

### 常见问题

1. **内存不足**
   - 减少batch_size
   - 使用Base模型而非Large模型
   - 启用梯度检查点

2. **CUDA错误**
   - 检查CUDA版本兼容性
   - 更新PyTorch版本

3. **分词问题**
   - 确保文本编码为UTF-8
   - 检查阿拉伯语文本方向

### 环境检查
```python
import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
```

## 📚 更多资源

- [AraBERTv2 论文](https://arxiv.org/abs/2003.00104)
- [Hugging Face 模型页面](https://huggingface.co/aubmindlab/bert-base-arabertv2)
- [项目GitHub](https://github.com/aub-mind/arabert)

## 🤝 贡献

欢迎提交Issue和Pull Request来改进项目！

---

**注意**: 首次运行时会自动下载模型文件（约500MB），请确保网络连接稳定。