"""
评估模块
评估训练好的AraBERTv2地址解析模型
"""

import os
import json
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report as seq_classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from configs.config import *
from src.model.arabertv2_ner import AraBERTv2NER, AraBERTv2Tokenizer
from src.training.train import AddressNERDataset


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() and DEVICE_CONFIG["use_cuda"] else "cpu")
        
        # 加载模型配置
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # 初始化模型和分词器
        self.tokenizer = AraBERTv2Tokenizer(self.config["model_name"])
        self.model = AraBERTv2NER(
            model_name=self.config["model_name"],
            num_labels=self.config["num_labels"]
        ).to(self.device)
        
        # 加载模型权重
        model_weights_path = os.path.join(model_path, "pytorch_model.bin")
        self.model.load_state_dict(torch.load(model_weights_path, map_location=self.device))
        self.model.eval()
        
        self.id_to_label = self.config["id_to_label"]
        self.label_to_id = self.config["label_to_id"]
        
        print(f"模型已从 {model_path} 加载")
    
    def predict_text(self, text: str) -> Tuple[List[str], List[str]]:
        """
        对单个文本进行预测
        
        Args:
            text: 输入文本
            
        Returns:
            (tokens, predicted_labels) 元组
        """
        # 简单分词
        tokens = text.split()
        
        # 编码
        encoding = self.tokenizer.tokenizer(
            tokens,
            truncation=True,
            padding=True,
            max_length=MODEL_CONFIG["max_length"],
            return_tensors="pt",
            is_split_into_words=True
        )
        
        # 移动到设备
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        
        # 预测
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs["logits"], dim=-1)
        
        # 对齐预测结果
        word_ids = encoding.word_ids()
        predicted_labels = []
        previous_word_idx = None
        
        for i, word_idx in enumerate(word_ids):
            if word_idx is not None and word_idx != previous_word_idx:
                if word_idx < len(tokens):
                    pred_id = predictions[0][i].item()
                    predicted_labels.append(self.id_to_label.get(str(pred_id), "O"))
            previous_word_idx = word_idx
        
        # 确保标签数量与token数量一致
        while len(predicted_labels) < len(tokens):
            predicted_labels.append("O")
        predicted_labels = predicted_labels[:len(tokens)]
        
        return tokens, predicted_labels
    
    def extract_entities(self, tokens: List[str], labels: List[str]) -> List[Dict]:
        """
        从BIO标签中提取实体
        
        Args:
            tokens: token列表
            labels: 标签列表
            
        Returns:
            实体列表
        """
        entities = []
        current_entity = None
        
        for i, (token, label) in enumerate(zip(tokens, labels)):
            if label.startswith("B-"):
                # 开始新实体
                if current_entity:
                    entities.append(current_entity)
                
                entity_type = label[2:]
                current_entity = {
                    "text": token,
                    "label": entity_type,
                    "start": i,
                    "end": i
                }
            
            elif label.startswith("I-") and current_entity:
                # 继续当前实体
                entity_type = label[2:]
                if entity_type == current_entity["label"]:
                    current_entity["text"] += " " + token
                    current_entity["end"] = i
                else:
                    # 实体类型不匹配，结束当前实体
                    entities.append(current_entity)
                    current_entity = None
            
            else:
                # O标签或其他情况，结束当前实体
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        # 处理最后一个实体
        if current_entity:
            entities.append(current_entity)
        
        return entities
    
    def evaluate_dataset(self, test_data_path: str) -> Dict:
        """
        评估整个数据集
        
        Args:
            test_data_path: 测试数据路径
            
        Returns:
            评估结果
        """
        # 加载测试数据
        with open(test_data_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        all_true_labels = []
        all_pred_labels = []
        all_true_entities = []
        all_pred_entities = []
        
        print(f"评估 {len(test_data)} 个样本...")
        
        for item in test_data:
            tokens = item["tokens"]
            true_labels = item["labels"]
            
            # 预测
            _, pred_labels = self.predict_text(" ".join(tokens))
            
            # 确保长度一致
            min_len = min(len(true_labels), len(pred_labels))
            true_labels = true_labels[:min_len]
            pred_labels = pred_labels[:min_len]
            tokens = tokens[:min_len]
            
            all_true_labels.append(true_labels)
            all_pred_labels.append(pred_labels)
            
            # 提取实体
            true_entities = self.extract_entities(tokens, true_labels)
            pred_entities = self.extract_entities(tokens, pred_labels)
            
            all_true_entities.append(true_entities)
            all_pred_entities.append(pred_entities)
        
        # 计算序列级别的指标
        seq_f1 = f1_score(all_true_labels, all_pred_labels)
        seq_precision = precision_score(all_true_labels, all_pred_labels)
        seq_recall = recall_score(all_true_labels, all_pred_labels)
        
        # 生成分类报告
        seq_report = seq_classification_report(all_true_labels, all_pred_labels)
        
        results = {
            "sequence_f1": seq_f1,
            "sequence_precision": seq_precision,
            "sequence_recall": seq_recall,
            "classification_report": seq_report,
            "num_samples": len(test_data)
        }
        
        return results
    
    def demo_prediction(self, texts: List[str]):
        """
        演示预测功能
        
        Args:
            texts: 要预测的文本列表
        """
        print("\\n=== 预测演示 ===")
        
        for i, text in enumerate(texts, 1):
            print(f"\\n示例 {i}:")
            print(f"输入: {text}")
            
            tokens, labels = self.predict_text(text)
            entities = self.extract_entities(tokens, labels)
            
            print("预测结果:")
            for token, label in zip(tokens, labels):
                print(f"  {token} -> {label}")
            
            print("提取的实体:")
            if entities:
                for entity in entities:
                    print(f"  {entity['text']} ({entity['label']})")
            else:
                print("  未检测到实体")
            
            print("-" * 50)


def main():
    """主函数"""
    # 检查模型是否存在
    model_path = os.path.join(MODEL_OUTPUT_DIR, "best_model_epoch_1")  # 根据实际情况调整
    
    if not os.path.exists(model_path):
        print(f"模型路径不存在: {model_path}")
        print("请先训练模型: python src/training/train.py")
        return
    
    # 创建评估器
    evaluator = ModelEvaluator(model_path)
    
    # 测试数据路径
    test_data_path = os.path.join(PROCESSED_DATA_DIR, "processed_sample.json")
    
    if os.path.exists(test_data_path):
        # 评估数据集
        results = evaluator.evaluate_dataset(test_data_path)
        
        print("\\n=== 评估结果 ===")
        print(f"序列级别 F1: {results['sequence_f1']:.4f}")
        print(f"序列级别 Precision: {results['sequence_precision']:.4f}")
        print(f"序列级别 Recall: {results['sequence_recall']:.4f}")
        print(f"样本数量: {results['num_samples']}")
        
        print("\\n详细分类报告:")
        print(results['classification_report'])
    
    # 演示预测
    demo_texts = [
        "شارع الملك فهد، حي الملز، الرياض، المملكة العربية السعودية",
        "طريق الأمير محمد بن عبدالعزيز، الدمام، المنطقة الشرقية",
        "مبنى رقم 123، شارع التحلية، جدة 21411"
    ]
    
    evaluator.demo_prediction(demo_texts)


if __name__ == "__main__":
    main()