#!/usr/bin/env python3
"""
AraBERTv2 阿拉伯语地址解析项目演示脚本
"""

import os
import sys
from src.data_processing.preprocess import ArabicAddressPreprocessor
from src.utils.helpers import set_seed, setup_logging, get_device_info
from configs.config import *


def main():
    """主演示函数"""
    print("=" * 60)
    print("AraBERTv2 阿拉伯语地址解析项目")
    print("=" * 60)
    
    # 设置随机种子
    set_seed(42)
    
    # 设置日志
    setup_logging()
    
    # 显示设备信息
    device_info = get_device_info()
    print("\\n设备信息:")
    for key, value in device_info.items():
        print(f"  {key}: {value}")
    
    # 创建必要的目录
    directories = [
        DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, SAMPLE_DATA_DIR,
        OUTPUT_DIR, MODEL_OUTPUT_DIR, LOG_DIR
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print(f"\\n项目目录已创建:")
    for directory in directories:
        print(f"  {directory}")
    
    # 数据预处理演示
    print("\\n" + "=" * 40)
    print("数据预处理演示")
    print("=" * 40)
    
    preprocessor = ArabicAddressPreprocessor()
    
    # 创建示例数据
    print("\\n创建示例数据...")
    processed_data = preprocessor.create_sample_dataset()
    
    # 显示处理结果
    print(f"\\n处理完成! 共 {len(processed_data)} 个样本")
    
    # 显示第一个样本的详细信息
    if processed_data:
        sample = processed_data[0]
        print("\\n第一个样本详情:")
        print(f"原文: {sample['original_text']}")
        print("分词和标签:")
        for token, label in zip(sample['tokens'], sample['labels']):
            print(f"  {token:15} -> {label}")
    
    # 显示配置信息
    print("\\n" + "=" * 40)
    print("项目配置")
    print("=" * 40)
    
    print("\\n模型配置:")
    for key, value in MODEL_CONFIG.items():
        print(f"  {key}: {value}")
    
    print("\\n实体标签:")
    for label, id in ENTITY_LABELS.items():
        print(f"  {label}: {id}")
    
    # 使用说明
    print("\\n" + "=" * 40)
    print("下一步操作")
    print("=" * 40)
    
    print("\\n1. 安装依赖:")
    print("   pip install -r requirements.txt")
    
    print("\\n2. 训练模型:")
    print("   python src/training/train.py")
    
    print("\\n3. 评估模型:")
    print("   python src/training/evaluate.py")
    
    print("\\n4. 查看演示notebook:")
    print("   jupyter notebook notebooks/demo.ipynb")
    
    print("\\n项目初始化完成! 🎉")


if __name__ == "__main__":
    main()