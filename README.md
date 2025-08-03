# AraBERTv2 阿拉伯语地址解析项目

基于AraBERTv2-large模型,训练一个用于阿拉伯语地址解析的NER（命名实体识别）模型。

## 项目结构

```
araBertv2/
├── data/                   # 数据目录
│   ├── raw/               # 原始数据
│   ├── processed/         # 预处理后的数据
│   └── sample/            # 示例数据
├── src/                   # 源代码
│   ├── data_processing/   # 数据处理模块
│   ├── model/            # 模型定义
│   ├── training/         # 训练相关代码
│   └── utils/            # 工具函数
├── configs/              # 配置文件
├── notebooks/            # Jupyter notebooks
├── outputs/              # 输出目录
│   ├── models/           # 训练好的模型
│   └── logs/             # 日志文件
└── tests/                # 测试文件
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

1. 准备数据：将阿拉伯语地址数据放入 `data/raw/` 目录
2. 数据预处理：运行 `python src/data_processing/preprocess.py`
3. 训练模型：运行 `python src/training/train.py`
4. 评估模型：运行 `python src/training/evaluate.py`

## 地址实体标签

- STREET: 街道名称
- BUILDING: 建筑物名称/编号
- DISTRICT: 区域/地区
- CITY: 城市
- COUNTRY: 国家
- POSTAL_CODE: 邮政编码