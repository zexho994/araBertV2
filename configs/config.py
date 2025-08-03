"""
配置文件 - AraBERTv2 阿拉伯语地址解析项目
"""

import os

# 基础路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# 数据路径
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
SAMPLE_DATA_DIR = os.path.join(DATA_DIR, "sample")

# 输出路径
MODEL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "models")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

# 模型配置
MODEL_CONFIG = {
    "model_name": "aubmindlab/bert-base-arabertv2",  # AraBERTv2 base model
    "model_name_large": "aubmindlab/bert-large-arabertv2",  # AraBERTv2 large model
    "max_length": 512,
    "num_labels": 13,  # B-/I- for 6 entity types + O
}

# 训练配置
TRAINING_CONFIG = {
    "batch_size": 16,
    "learning_rate": 2e-5,
    "num_epochs": 10,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "save_steps": 1000,
    "eval_steps": 500,
    "logging_steps": 100,
}

# 实体标签配置
ENTITY_LABELS = {
    "O": 0,
    "B-STREET": 1,
    "I-STREET": 2,
    "B-BUILDING": 3,
    "I-BUILDING": 4,
    "B-DISTRICT": 5,
    "I-DISTRICT": 6,
    "B-CITY": 7,
    "I-CITY": 8,
    "B-COUNTRY": 9,
    "I-COUNTRY": 10,
    "B-POSTAL_CODE": 11,
    "I-POSTAL_CODE": 12,
}

# 标签到ID的映射
LABEL_TO_ID = ENTITY_LABELS
ID_TO_LABEL = {v: k for k, v in ENTITY_LABELS.items()}

# 设备配置
DEVICE_CONFIG = {
    "use_cuda": True,
    "cuda_device": 0,
}