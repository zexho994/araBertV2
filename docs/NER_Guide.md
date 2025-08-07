# 项目设计

```Textile
ner\_cli.py                    # NER CLI主入口文件
ner\_data/                     # NER数据目录（类似dapt\_data）
├── configs/                  # 配置文件目录
│   ├── countries/           # 各国配置文件
│   │   ├── uae.json        # 阿联酋配置
│   │   ├── saudi.json      # 沙特配置
│   │   └── egypt.json      # 埃及配置
│   └── templates/           # 配置模板
│       ├── default.json    # 默认NER配置模板
│       └── address\_ner.json # 地址解析专用模板
├── training\_data/           # 训练数据目录
│   ├── uae/                # 阿联酋训练数据
│   │   ├── train.jsonl     # 训练集
│   │   ├── validation.jsonl # 验证集
│   │   └── test.jsonl      # 测试集
│   └── \[country]/          # 其他国家数据
├── models/                  # 模型存储目录
│   └── uae/                # 阿联酋模型
│       ├── checkpoint-\*/   # 训练检查点
│       ├── final\_model/    # 最终模型
│       └── training\_config.json
├── evaluation/              # 评估结果目录
│   └── uae/
│       ├── reports/        # 评估报告
│       └── predictions/    # 预测结果
└── logs/                   # 日志目录
└── cli/               # CLI日志
src/
└── ner/                    # NER模块源码
├── cli/               # CLI命令实现
├── config/            # 配置管理
├── data/              # 数据处理
├── models/            # 模型管理
├── training/          # 训练引擎
├── evaluation/        # 评估模块
└── utils/             # 工具函数
```

# 使用示例

## 训练UAE地址解析模型

python ner\_cli.py train --country uae --epochs 20 --batch-size 32

## 评估模型

python ner\_cli.py evaluate --model uae\_v1.0 --test-data uae\_test.jsonl

## 预测地址实体

python ner\_cli.py predict --model uae\_v1.0 --text "شارع الشيخ زايد، دبي، الإمارات"

## 创建新国家配置

python ner\_cli.py config create --country saudi --template address\_ner

## 验证训练数据

python ner\_cli.py data validate --country uae --input-file training\_data.jsonl

# 国家配置文件设计

```json
{
  "country": {
    "code": "uae",
    "name": "United Arab Emirates",
    "language": "ar",
    "script": "arabic",
    "region": "middle_east"
  },
  "model": {
    "base_model": "aubmindlab/bert-base-arabertv2",
    "model_type": "bert_for_token_classification",
    "max_length": 256,
    "num_labels": 23,
    "dropout_rate": 0.1,
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1
  },
  "training": {
    "learning_rate": 2e-5,
    "batch_size": 32,
    "gradient_accumulation_steps": 1,
    "epochs": 20,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "adam_epsilon": 1e-8,
    "max_grad_norm": 1.0,
    "scheduler_type": "linear",
    "save_strategy": "steps",
    "save_steps": 500,
    "evaluation_strategy": "steps",
    "eval_steps": 500,
    "logging_steps": 100,
    "fp16": true,
    "dataloader_num_workers": 4,
    "seed": 42
  },
  "data": {
    "train_file": "data/ner/training_data/uae/train.jsonl",
    "validation_file": "data/ner/training_data/uae/validation.jsonl",
    "test_file": "data/ner/training_data/uae/test.jsonl",
    "data_format": "jsonl",
    "text_column": "text",
    "labels_column": "labels",
    "preprocessing": {
      "clean_text": true,
      "normalize_arabic": true,
      "remove_diacritics": false,
      "handle_mixed_script": true,
      "max_length": 256,
      "padding": "max_length",
      "truncation": true,
      "return_attention_mask": true,
      "return_token_type_ids": false,
      "label_all_tokens": false
    }
  },
  "labels": {
    "label_names": [
      "O",
      "B-COUNTRY", "I-COUNTRY",
      "B-EMIRATE", "I-EMIRATE",
      "B-CITY", "I-CITY",
      "B-SUB_AREA", "I-SUB_AREA",
      "B-COMPOUND", "I-COMPOUND",
      "B-STREET", "I-STREET",
      "B-BUILDING", "I-BUILDING",
      "B-HOUSE_NUMBER", "I-HOUSE_NUMBER",
      "B-LANDMARK", "I-LANDMARK",
      "B-POSTAL_CODE", "I-POSTAL_CODE",
      "B-MAKANI_NUMBER", "I-MAKANI_NUMBER"
    ],
    "label_mapping": {
      "O": 0,
      "B-COUNTRY": 1, "I-COUNTRY": 2,
      "B-EMIRATE": 3, "I-EMIRATE": 4,
      "B-CITY": 5, "I-CITY": 6,
      "B-SUB_AREA": 7, "I-SUB_AREA": 8,
      "B-COMPOUND": 9, "I-COMPOUND": 10,
      "B-STREET": 11, "I-STREET": 12,
      "B-BUILDING": 13, "I-BUILDING": 14,
      "B-HOUSE_NUMBER": 15, "I-HOUSE_NUMBER": 16,
      "B-LANDMARK": 17, "I-LANDMARK": 18,
      "B-POSTAL_CODE": 19, "I-POSTAL_CODE": 20,
      "B-MAKANI_NUMBER": 21, "I-MAKANI_NUMBER": 22
    }
  },
  "evaluation": {
    "metrics": ["accuracy", "f1", "precision", "recall", "seqeval"],
    "average": "weighted",
    "save_predictions": true,
    "save_misclassified": true,
    "confusion_matrix": true,
    "per_label_metrics": true
  },
  "output": {
    "output_dir": "data/ner/models/uae",
    "logs_dir": "data/ner/logs/uae",
    "cache_dir": "cache/uae",
    "save_model": true,
    "save_tokenizer": true,
    "save_config": true,
    "save_total_limit": 3,
    "load_best_model_at_end": true
  },
  "hardware": {
    "device": "auto",
    "gpu_ids": [0],
    "max_memory_gb": 16,
    "gradient_checkpointing": true,
    "distributed": false,
    "use_fast_tokenizer": true
  },
  "logging": {
    "level": "INFO",
    "log_file": "uae_ner_training.log",
    "progress_bar": true,
    "wandb": {
      "enabled": false,
      "project": "ner-address-parsing",
      "entity": "your-team"
    },
    "tensorboard": {
      "enabled": true,
      "log_dir": "data/ner/logs/uae/tensorboard"
    }
  }
}
```

# 4. 配置项详细说明

## country 部分：

* code : 国家代码，用于标识和组织文件

* name : 国家全名

* language : 主要语言代码

* script : 文字系统

* region : 地理区域

## model 部分：

* base\_model : 基础预训练模型

* model\_type : 模型类型（token分类）

* max\_length : 最大序列长度

* num\_labels : 标签数量

* dropout\_rate : Dropout比率

## training 部分：

* learning\_rate : 学习率

* batch\_size : 批次大小

* epochs : 训练轮数

* warmup\_steps : 预热步数

* weight\_decay : 权重衰减

* scheduler\_type : 学习率调度器类型

## data 部分：

* train\_file/validation\_file/test\_file : 数据文件路径

* data\_format : 数据格式（jsonl）

* text\_column/labels\_column : 文本和标签列名

* preprocessing : 数据预处理配置

## labels 部分：

* label\_names : 所有标签名称列表

* label\_mapping : 标签到数字的映射

## evaluation 部分：

* metrics : 评估指标列表

* save\_predictions : 是否保存预测结果

* confusion\_matrix : 是否生成混淆矩阵

