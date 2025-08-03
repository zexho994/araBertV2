"""
数据预处理模块
处理阿拉伯语地址数据，转换为NER训练格式
"""

import os
import json
import pandas as pd
from typing import List, Tuple, Dict, Any
import re
from arabic_reshaper import reshape
from bidi.algorithm import get_display

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from configs.config import *


class ArabicAddressPreprocessor:
    """阿拉伯语地址数据预处理器"""
    
    def __init__(self):
        self.entity_labels = ENTITY_LABELS
        self.label_to_id = LABEL_TO_ID
        
    def clean_arabic_text(self, text: str) -> str:
        """
        清理阿拉伯语文本
        
        Args:
            text: 原始文本
            
        Returns:
            清理后的文本
        """
        if not text:
            return ""
        
        # 移除多余的空格
        text = re.sub(r'\s+', ' ', text.strip())
        
        # 移除特殊字符（保留阿拉伯语字符、数字、基本标点）
        text = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF\s\d\.,،؛؟!]', '', text)
        
        return text
    
    def tokenize_arabic_text(self, text: str) -> List[str]:
        """
        阿拉伯语文本分词
        
        Args:
            text: 输入文本
            
        Returns:
            分词结果
        """
        # 简单的基于空格的分词
        tokens = text.split()
        return [token.strip() for token in tokens if token.strip()]
    
    def create_sample_data(self) -> List[Dict[str, Any]]:
        """
        创建示例数据
        
        Returns:
            示例数据列表
        """
        sample_data = [
            {
                "text": "شارع الملك فهد، حي الملز، الرياض، المملكة العربية السعودية",
                "entities": [
                    {"start": 0, "end": 13, "label": "STREET", "text": "شارع الملك فهد"},
                    {"start": 16, "end": 24, "label": "DISTRICT", "text": "حي الملز"},
                    {"start": 27, "end": 34, "label": "CITY", "text": "الرياض"},
                    {"start": 37, "end": 62, "label": "COUNTRY", "text": "المملكة العربية السعودية"}
                ]
            },
            {
                "text": "طريق الأمير محمد بن عبدالعزيز، الدمام، المنطقة الشرقية",
                "entities": [
                    {"start": 0, "end": 30, "label": "STREET", "text": "طريق الأمير محمد بن عبدالعزيز"},
                    {"start": 33, "end": 39, "label": "CITY", "text": "الدمام"},
                    {"start": 42, "end": 57, "label": "DISTRICT", "text": "المنطقة الشرقية"}
                ]
            },
            {
                "text": "مبنى رقم 123، شارع التحلية، جدة 21411",
                "entities": [
                    {"start": 0, "end": 11, "label": "BUILDING", "text": "مبنى رقم 123"},
                    {"start": 14, "end": 26, "label": "STREET", "text": "شارع التحلية"},
                    {"start": 29, "end": 32, "label": "CITY", "text": "جدة"},
                    {"start": 33, "end": 38, "label": "POSTAL_CODE", "text": "21411"}
                ]
            }
        ]
        return sample_data
    
    def convert_to_bio_format(self, text: str, entities: List[Dict]) -> Tuple[List[str], List[str]]:
        """
        将实体标注转换为BIO格式
        
        Args:
            text: 原始文本
            entities: 实体列表
            
        Returns:
            (tokens, labels) 元组
        """
        tokens = self.tokenize_arabic_text(text)
        labels = ["O"] * len(tokens)
        
        # 为每个实体分配标签
        for entity in entities:
            entity_text = entity["text"]
            entity_label = entity["label"]
            entity_tokens = self.tokenize_arabic_text(entity_text)
            
            # 在tokens中找到实体的位置
            for i in range(len(tokens) - len(entity_tokens) + 1):
                if tokens[i:i+len(entity_tokens)] == entity_tokens:
                    # 分配B-和I-标签
                    labels[i] = f"B-{entity_label}"
                    for j in range(1, len(entity_tokens)):
                        if i + j < len(labels):
                            labels[i + j] = f"I-{entity_label}"
                    break
        
        return tokens, labels
    
    def process_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        处理数据，转换为训练格式
        
        Args:
            data: 原始数据
            
        Returns:
            处理后的数据
        """
        processed_data = []
        
        for item in data:
            text = self.clean_arabic_text(item["text"])
            entities = item.get("entities", [])
            
            tokens, labels = self.convert_to_bio_format(text, entities)
            
            # 转换标签为ID
            label_ids = [self.label_to_id.get(label, 0) for label in labels]
            
            processed_item = {
                "tokens": tokens,
                "labels": labels,
                "label_ids": label_ids,
                "original_text": item["text"]
            }
            
            processed_data.append(processed_item)
        
        return processed_data
    
    def save_processed_data(self, data: List[Dict[str, Any]], output_path: str):
        """
        保存处理后的数据
        
        Args:
            data: 处理后的数据
            output_path: 输出路径
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"数据已保存到: {output_path}")
    
    def create_sample_dataset(self):
        """创建示例数据集"""
        # 创建目录
        os.makedirs(SAMPLE_DATA_DIR, exist_ok=True)
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        
        # 生成示例数据
        sample_data = self.create_sample_data()
        
        # 保存原始示例数据
        sample_path = os.path.join(SAMPLE_DATA_DIR, "sample_addresses.json")
        with open(sample_path, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)
        
        # 处理数据
        processed_data = self.process_data(sample_data)
        
        # 保存处理后的数据
        processed_path = os.path.join(PROCESSED_DATA_DIR, "processed_sample.json")
        self.save_processed_data(processed_data, processed_path)
        
        print(f"示例数据集已创建:")
        print(f"- 原始数据: {sample_path}")
        print(f"- 处理后数据: {processed_path}")
        
        return processed_data


def main():
    """主函数"""
    preprocessor = ArabicAddressPreprocessor()
    
    # 创建示例数据集
    processed_data = preprocessor.create_sample_dataset()
    
    # 打印统计信息
    print(f"\n数据统计:")
    print(f"- 总样本数: {len(processed_data)}")
    
    # 统计标签分布
    label_counts = {}
    for item in processed_data:
        for label in item["labels"]:
            label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f"- 标签分布:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()