# NER CLI Predict 命令使用指南

本文档详细介绍如何使用 NER CLI 的 `predict` 命令进行文本预测，以及如何分析预测结果。

## 目录

1. [基本语法和参数说明](#基本语法和参数说明)
2. [使用示例](#使用示例)
3. [输出格式详解](#输出格式详解)
4. [预测结果分析](#预测结果分析)
5. [置信度和实体提取分析](#置信度和实体提取分析)
6. [常见问题和故障排除](#常见问题和故障排除)
7. [高级用法和最佳实践](#高级用法和最佳实践)

## 基本语法和参数说明

### 命令语法

```bash
python ner_cli.py predict [参数]
```

### 必需参数

- `--model-path, -m`: 训练好的模型路径
- `--text, -t` 或 `--file, -f`: 要分析的文本（二选一）
  - `--text`: 直接输入文本字符串
  - `--file`: 包含文本的文件路径

### 可选参数

- `--country`: 国家配置（用于加载特定的配置设置）
- `--output-file`: 输出文件路径（如果不指定，结果将输出到控制台）
- `--output-format`: 输出格式，可选值：
  - `json`（默认）: JSON 格式输出
  - `text`: 纯文本格式输出
  - `conll`: CoNLL 格式输出
- `--confidence-threshold`: 置信度阈值（默认：0.5）

## 使用示例

### 1. 单文本预测

#### 基本用法

```bash
# 使用 JSON 格式输出（默认）
python ner_cli.py predict \
  --model-path ./data/ner/models/uae_model \
  --text "123 Sheikh Zayed Road, Dubai, UAE"
```

#### 指定输出格式

```bash
# 使用文本格式输出
python ner_cli.py predict \
  --model-path ./data/ner/models/uae_model \
  --text "123 Sheikh Zayed Road, Dubai, UAE" \
  --output-format text

# 使用 CoNLL 格式输出
python ner_cli.py predict \
  --model-path ./data/ner/models/uae_model \
  --text "123 Sheikh Zayed Road, Dubai, UAE" \
  --output-format conll
```

#### 设置置信度阈值

```bash
# 只显示置信度大于 0.8 的预测结果
python ner_cli.py predict \
  --model-path ./data/ner/models/uae_model \
  --text "123 Sheikh Zayed Road, Dubai, UAE" \
  --confidence-threshold 0.8
```

### 2. 批量文件预测

#### 从文件读取文本

```bash
# 从文件读取文本进行预测
python ner_cli.py predict \
  --model-path ./data/ner/models/uae_model \
  --file ./data/test_texts.txt
```

#### 保存结果到文件

```bash
# 将预测结果保存到文件
python ner_cli.py predict \
  --model-path ./data/ner/models/uae_model \
  --file ./data/test_texts.txt \
  --output-file ./results/predictions.json
```

### 3. 使用国家配置

```bash
# 使用特定国家配置
python ner_cli.py predict \
  --model-path ./data/ner/models/uae_model \
  --country uae \
  --text "123 Sheikh Zayed Road, Dubai, UAE"
```

## 输出格式详解

### 1. JSON 格式（默认）

JSON 格式提供最详细的预测信息：

```json
{
  "text": "123 Sheikh Zayed Road, Dubai, UAE",
  "tokens": ["123", "Sheikh", "Zayed", "Road", ",", "Dubai", ",", "UAE"],
  "labels": ["B-STREET_NUMBER", "B-STREET_NAME", "I-STREET_NAME", "I-STREET_NAME", "O", "B-CITY", "O", "B-COUNTRY"],
  "confidences": [0.95, 0.92, 0.89, 0.87, 0.99, 0.94, 0.98, 0.96],
  "entities": [
    {
      "text": "123",
      "label": "STREET_NUMBER",
      "start": 0,
      "end": 3,
      "confidence": 0.95
    },
    {
      "text": "Sheikh Zayed Road",
      "label": "STREET_NAME",
      "start": 4,
      "end": 21,
      "confidence": 0.89
    },
    {
      "text": "Dubai",
      "label": "CITY",
      "start": 23,
      "end": 28,
      "confidence": 0.94
    },
    {
      "text": "UAE",
      "label": "COUNTRY",
      "start": 30,
      "end": 33,
      "confidence": 0.96
    }
  ]
}
```

### 2. Text 格式

文本格式提供人类可读的输出：

```
Text: 123 Sheikh Zayed Road, Dubai, UAE

Entities:
- STREET_NUMBER: "123" (confidence: 0.95)
- STREET_NAME: "Sheikh Zayed Road" (confidence: 0.89)
- CITY: "Dubai" (confidence: 0.94)
- COUNTRY: "UAE" (confidence: 0.96)

Tokens and Labels:
123 -> B-STREET_NUMBER (0.95)
Sheikh -> B-STREET_NAME (0.92)
Zayed -> I-STREET_NAME (0.89)
Road -> I-STREET_NAME (0.87)
, -> O (0.99)
Dubai -> B-CITY (0.94)
, -> O (0.98)
UAE -> B-COUNTRY (0.96)
```

### 3. CoNLL 格式

CoNLL 格式适用于进一步的数据处理：

```
123	B-STREET_NUMBER
Sheikh	B-STREET_NAME
Zayed	I-STREET_NAME
Road	I-STREET_NAME
,	O
Dubai	B-CITY
,	O
UAE	B-COUNTRY
```

## 预测结果分析

### 结果结构解析

#### JSON 输出字段含义

1. **text**: 原始输入文本
2. **tokens**: 分词后的词元列表
3. **labels**: 每个词元对应的 BIO 标签
4. **confidences**: 每个词元预测的置信度分数
5. **entities**: 提取的实体列表，包含：
   - `text`: 实体文本
   - `label`: 实体类型
   - `start`: 在原文中的起始位置
   - `end`: 在原文中的结束位置
   - `confidence`: 实体的平均置信度

#### BIO 标签说明

- **B-**: 实体的开始（Begin）
- **I-**: 实体的内部（Inside）
- **O**: 非实体（Outside）

例如："Sheikh Zayed Road" 被标记为：
- Sheikh: B-STREET_NAME（街道名称的开始）
- Zayed: I-STREET_NAME（街道名称的内部）
- Road: I-STREET_NAME（街道名称的内部）

### 实体提取过程

模型通过以下步骤提取实体：

1. **分词**: 将文本分解为词元
2. **标签预测**: 为每个词元预测 BIO 标签
3. **置信度计算**: 计算每个预测的置信度
4. **实体组装**: 将连续的 B- 和 I- 标签组合成完整实体
5. **过滤**: 根据置信度阈值过滤低置信度的预测

## 置信度和实体提取分析

### 置信度分析

#### 置信度含义

- **高置信度 (>0.9)**: 模型非常确信的预测
- **中等置信度 (0.7-0.9)**: 模型较为确信的预测
- **低置信度 (0.5-0.7)**: 模型不太确信的预测
- **极低置信度 (<0.5)**: 可能的误预测

#### 置信度阈值设置建议

```bash
# 高精度场景（减少误报）
--confidence-threshold 0.8

# 平衡场景（默认）
--confidence-threshold 0.5

# 高召回场景（减少漏报）
--confidence-threshold 0.3
```

### 实体质量评估

#### 评估指标

1. **实体级别置信度**: 实体内所有词元置信度的平均值
2. **边界准确性**: B- 和 I- 标签的一致性
3. **实体完整性**: 实体是否被完整识别

#### 质量判断标准

```python
# 高质量实体
if entity['confidence'] > 0.8:
    print(f"高质量实体: {entity['text']} ({entity['label']})")

# 需要人工验证的实体
elif entity['confidence'] > 0.5:
    print(f"需要验证: {entity['text']} ({entity['label']})")

# 可能的误识别
else:
    print(f"可能误识别: {entity['text']} ({entity['label']})")
```

## 常见问题和故障排除

### 1. 模型加载失败

**问题**: `FileNotFoundError: Model not found`

**解决方案**:
```bash
# 检查模型路径是否正确
ls -la ./data/ner/models/uae_model

# 确保模型文件完整
ls -la ./data/ner/models/uae_model/
# 应该包含: config.json, pytorch_model.bin, tokenizer.json 等
```

### 2. 内存不足

**问题**: `CUDA out of memory` 或系统内存不足

**解决方案**:
```bash
# 处理较短的文本
# 或者使用 CPU 模式（在模型配置中设置）
export CUDA_VISIBLE_DEVICES=""
```

### 3. 分词问题

**问题**: 文本分词结果不理想

**解决方案**:
- 确保使用与训练时相同的分词器
- 检查文本编码（建议使用 UTF-8）
- 预处理文本，移除特殊字符

### 4. 置信度异常

**问题**: 所有预测置信度都很低

**可能原因**:
- 输入文本与训练数据分布差异较大
- 模型未充分训练
- 文本预处理不当

**解决方案**:
- 检查输入文本格式
- 降低置信度阈值进行测试
- 重新训练模型或使用更多训练数据

### 5. 输出格式问题

**问题**: 输出格式不符合预期

**解决方案**:
```bash
# 检查输出格式参数
--output-format json  # 确保格式正确

# 检查输出文件权限
touch ./results/test.json  # 测试文件写入权限
```

## 高级用法和最佳实践

### 1. 批量处理优化

#### 处理大文件

```bash
# 将大文件分割成小块处理
split -l 1000 large_file.txt chunk_

# 批量处理
for file in chunk_*; do
    python ner_cli.py predict \
        --model-path ./models/uae_model \
        --file $file \
        --output-file results_$file.json
done
```

#### 并行处理脚本示例

```python
#!/usr/bin/env python3
import subprocess
import concurrent.futures
import os

def process_file(file_path):
    """处理单个文件"""
    output_file = f"results_{os.path.basename(file_path)}.json"
    cmd = [
        "python", "ner_cli.py", "predict",
        "--model-path", "./models/uae_model",
        "--file", file_path,
        "--output-file", output_file
    ]
    subprocess.run(cmd)
    return output_file

# 并行处理多个文件
files = ["file1.txt", "file2.txt", "file3.txt"]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_file, files))
```

### 2. 结果后处理

#### 实体去重和合并

```python
import json

def merge_entities(entities):
    """合并重叠的实体"""
    merged = []
    entities = sorted(entities, key=lambda x: x['start'])
    
    for entity in entities:
        if not merged or entity['start'] >= merged[-1]['end']:
            merged.append(entity)
        else:
            # 合并重叠实体，保留置信度更高的
            if entity['confidence'] > merged[-1]['confidence']:
                merged[-1] = entity
    
    return merged

# 使用示例
with open('predictions.json', 'r') as f:
    result = json.load(f)
    
merged_entities = merge_entities(result['entities'])
print(f"原始实体数: {len(result['entities'])}")
print(f"合并后实体数: {len(merged_entities)}")
```

#### 实体验证和过滤

```python
def validate_entities(entities, min_confidence=0.7, min_length=2):
    """验证和过滤实体"""
    valid_entities = []
    
    for entity in entities:
        # 置信度过滤
        if entity['confidence'] < min_confidence:
            continue
            
        # 长度过滤
        if len(entity['text']) < min_length:
            continue
            
        # 自定义验证规则
        if entity['label'] == 'PHONE' and not is_valid_phone(entity['text']):
            continue
            
        valid_entities.append(entity)
    
    return valid_entities

def is_valid_phone(text):
    """验证电话号码格式"""
    import re
    phone_pattern = r'^[\+]?[1-9]?[0-9]{7,15}$'
    return bool(re.match(phone_pattern, text.replace(' ', '').replace('-', '')))
```

### 3. 性能监控

#### 预测性能统计

```python
import time
import json
from collections import defaultdict

def analyze_predictions(prediction_file):
    """分析预测结果统计信息"""
    with open(prediction_file, 'r') as f:
        result = json.load(f)
    
    stats = {
        'total_tokens': len(result['tokens']),
        'total_entities': len(result['entities']),
        'entity_types': defaultdict(int),
        'confidence_distribution': defaultdict(int),
        'avg_confidence': 0
    }
    
    # 统计实体类型
    for entity in result['entities']:
        stats['entity_types'][entity['label']] += 1
    
    # 统计置信度分布
    confidences = [entity['confidence'] for entity in result['entities']]
    if confidences:
        stats['avg_confidence'] = sum(confidences) / len(confidences)
        
        for conf in confidences:
            if conf >= 0.9:
                stats['confidence_distribution']['high'] += 1
            elif conf >= 0.7:
                stats['confidence_distribution']['medium'] += 1
            else:
                stats['confidence_distribution']['low'] += 1
    
    return stats

# 使用示例
stats = analyze_predictions('predictions.json')
print(f"总词元数: {stats['total_tokens']}")
print(f"总实体数: {stats['total_entities']}")
print(f"平均置信度: {stats['avg_confidence']:.3f}")
print(f"实体类型分布: {dict(stats['entity_types'])}")
print(f"置信度分布: {dict(stats['confidence_distribution'])}")
```

### 4. 集成到应用程序

#### Python API 调用示例

```python
import subprocess
import json
import tempfile

class NERPredictor:
    def __init__(self, model_path, confidence_threshold=0.5):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
    
    def predict_text(self, text):
        """预测单个文本"""
        cmd = [
            "python", "ner_cli.py", "predict",
            "--model-path", self.model_path,
            "--text", text,
            "--confidence-threshold", str(self.confidence_threshold),
            "--output-format", "json"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return json.loads(result.stdout)
        else:
            raise Exception(f"Prediction failed: {result.stderr}")
    
    def predict_batch(self, texts):
        """批量预测"""
        results = []
        for text in texts:
            try:
                result = self.predict_text(text)
                results.append(result)
            except Exception as e:
                print(f"Error processing text '{text[:50]}...': {e}")
                results.append(None)
        return results

# 使用示例
predictor = NERPredictor("./models/uae_model", confidence_threshold=0.7)

# 单文本预测
result = predictor.predict_text("123 Sheikh Zayed Road, Dubai")
print(f"发现 {len(result['entities'])} 个实体")

# 批量预测
texts = [
    "123 Sheikh Zayed Road, Dubai",
    "456 Al Wasl Road, Jumeirah",
    "789 Business Bay, Dubai"
]
results = predictor.predict_batch(texts)
print(f"处理了 {len([r for r in results if r])} 个文本")
```

### 5. 模型评估和比较

#### 预测结果评估

```python
def evaluate_predictions(predictions, ground_truth):
    """评估预测结果质量"""
    from sklearn.metrics import precision_recall_fscore_support
    
    # 提取实体进行比较
    pred_entities = set()
    true_entities = set()
    
    for pred in predictions['entities']:
        pred_entities.add((pred['start'], pred['end'], pred['label']))
    
    for true in ground_truth['entities']:
        true_entities.add((true['start'], true['end'], true['label']))
    
    # 计算精确率、召回率、F1值
    tp = len(pred_entities & true_entities)
    fp = len(pred_entities - true_entities)
    fn = len(true_entities - pred_entities)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }
```

## 总结

本指南涵盖了 NER CLI `predict` 命令的完整使用方法，从基本语法到高级应用。通过遵循这些最佳实践，您可以：

1. 高效地使用 predict 命令进行文本预测
2. 正确分析和解释预测结果
3. 优化预测性能和准确性
4. 集成到实际应用程序中
5. 监控和评估模型表现

建议新用户从基本示例开始，逐步掌握高级功能。如有问题，请参考故障排除部分或查看项目的其他文档。