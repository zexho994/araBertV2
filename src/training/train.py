"""
训练模块
使用AraBERTv2训练阿拉伯语地址解析模型
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm
import logging
from datetime import datetime

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from configs.config import *
from src.model.arabertv2_ner import AraBERTv2NER, AraBERTv2Tokenizer


# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AddressNERDataset(Dataset):
    """地址NER数据集"""
    
    def __init__(self, data: list, tokenizer: AraBERTv2Tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = item["tokens"]
        label_ids = item["label_ids"]
        
        # 分词和编码
        encoding = self.tokenizer.tokenizer(
            tokens,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
            is_split_into_words=True
        )
        
        # 对齐标签
        word_ids = encoding.word_ids()
        aligned_labels = []
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(-100)
            elif word_idx != previous_word_idx:
                if word_idx < len(label_ids):
                    aligned_labels.append(label_ids[word_idx])
                else:
                    aligned_labels.append(-100)
            else:
                aligned_labels.append(-100)
            previous_word_idx = word_idx
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(aligned_labels, dtype=torch.long)
        }


class AddressNERTrainer:
    """地址NER训练器"""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or MODEL_CONFIG["model_name_large"]
        self.device = torch.device("cuda" if torch.cuda.is_available() and DEVICE_CONFIG["use_cuda"] else "cpu")
        
        # 创建输出目录
        os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)
        
        # 初始化模型和分词器
        self.tokenizer = AraBERTv2Tokenizer(self.model_name)
        self.model = AraBERTv2NER(
            model_name=self.model_name,
            num_labels=MODEL_CONFIG["num_labels"]
        ).to(self.device)
        
        logger.info(f"模型已加载到设备: {self.device}")
        logger.info(f"使用模型: {self.model_name}")
    
    def load_data(self, data_path: str) -> list:
        """加载训练数据"""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"已加载 {len(data)} 个训练样本")
        return data
    
    def create_data_loader(self, data: list, batch_size: int, shuffle: bool = True) -> DataLoader:
        """创建数据加载器"""
        dataset = AddressNERDataset(
            data=data,
            tokenizer=self.tokenizer,
            max_length=MODEL_CONFIG["max_length"]
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn
        )
    
    def collate_fn(self, batch):
        """批处理函数"""
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    def train(self, train_data_path: str, val_data_path: str = None):
        """训练模型"""
        # 加载数据
        train_data = self.load_data(train_data_path)
        
        # 如果没有验证数据，使用训练数据的一部分
        if val_data_path:
            val_data = self.load_data(val_data_path)
        else:
            split_idx = int(len(train_data) * 0.8)
            val_data = train_data[split_idx:]
            train_data = train_data[:split_idx]
        
        # 创建数据加载器
        train_loader = self.create_data_loader(
            train_data, 
            TRAINING_CONFIG["batch_size"], 
            shuffle=True
        )
        val_loader = self.create_data_loader(
            val_data, 
            TRAINING_CONFIG["batch_size"], 
            shuffle=False
        )
        
        # 设置优化器和调度器
        optimizer = AdamW(
            self.model.parameters(),
            lr=TRAINING_CONFIG["learning_rate"],
            weight_decay=TRAINING_CONFIG["weight_decay"]
        )
        
        total_steps = len(train_loader) * TRAINING_CONFIG["num_epochs"]
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=TRAINING_CONFIG["warmup_steps"],
            num_training_steps=total_steps
        )
        
        # 训练循环
        best_f1 = 0.0
        global_step = 0
        
        for epoch in range(TRAINING_CONFIG["num_epochs"]):
            logger.info(f"开始第 {epoch + 1}/{TRAINING_CONFIG['num_epochs']} 轮训练")
            
            # 训练阶段
            self.model.train()
            total_loss = 0.0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
            
            for batch in progress_bar:
                # 移动数据到设备
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # 前向传播
                outputs = self.model(**batch)
                loss = outputs["loss"]
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                global_step += 1
                
                # 更新进度条
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "avg_loss": f"{total_loss / (global_step % len(train_loader) + 1):.4f}"
                })
                
                # 记录日志
                if global_step % TRAINING_CONFIG["logging_steps"] == 0:
                    logger.info(f"Step {global_step}, Loss: {loss.item():.4f}")
            
            # 验证阶段
            if val_loader:
                val_f1 = self.evaluate(val_loader)
                logger.info(f"Epoch {epoch + 1} 验证 F1: {val_f1:.4f}")
                
                # 保存最佳模型
                if val_f1 > best_f1:
                    best_f1 = val_f1
                    self.save_model(f"best_model_epoch_{epoch + 1}")
                    logger.info(f"保存最佳模型，F1: {best_f1:.4f}")
            
            # 定期保存模型
            if (epoch + 1) % 2 == 0:
                self.save_model(f"checkpoint_epoch_{epoch + 1}")
        
        logger.info(f"训练完成！最佳验证 F1: {best_f1:.4f}")
    
    def evaluate(self, data_loader: DataLoader) -> float:
        """评估模型"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="评估中"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(**batch)
                predictions = torch.argmax(outputs["logits"], dim=-1)
                
                # 收集预测和标签
                for i in range(predictions.size(0)):
                    pred = predictions[i].cpu().numpy()
                    label = batch["labels"][i].cpu().numpy()
                    attention_mask = batch["attention_mask"][i].cpu().numpy()
                    
                    # 只考虑有效的token
                    valid_indices = (attention_mask == 1) & (label != -100)
                    all_predictions.extend(pred[valid_indices])
                    all_labels.extend(label[valid_indices])
        
        # 计算F1分数
        f1 = f1_score(all_labels, all_predictions, average="weighted")
        
        return f1
    
    def save_model(self, model_name: str):
        """保存模型"""
        model_path = os.path.join(MODEL_OUTPUT_DIR, model_name)
        os.makedirs(model_path, exist_ok=True)
        
        # 保存模型权重
        torch.save(self.model.state_dict(), os.path.join(model_path, "pytorch_model.bin"))
        
        # 保存分词器
        self.tokenizer.tokenizer.save_pretrained(model_path)
        
        # 保存配置
        config = {
            "model_name": self.model_name,
            "num_labels": MODEL_CONFIG["num_labels"],
            "label_to_id": LABEL_TO_ID,
            "id_to_label": ID_TO_LABEL
        }
        
        with open(os.path.join(model_path, "config.json"), 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        logger.info(f"模型已保存到: {model_path}")


def main():
    """主函数"""
    # 检查数据文件是否存在 - 使用修正后的数据
    train_data_path = os.path.join(PROCESSED_DATA_DIR, "corrected_sample.json")
    
    if not os.path.exists(train_data_path):
        logger.error(f"训练数据文件不存在: {train_data_path}")
        logger.info("请确保已运行数据质量检查并生成修正后的数据")
        return

    logger.info("开始训练模型...")
    logger.info(f"训练数据路径: {train_data_path}")
    logger.info("使用修正后的高质量标注数据进行训练")
    
    # 创建训练器
    trainer = AddressNERTrainer()

    logger.info("创建训练器完成")
    
    # 开始训练
    trainer.train(train_data_path, train_data_path)


if __name__ == "__main__":
    main()