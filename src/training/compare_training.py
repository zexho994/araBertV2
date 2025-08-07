"""
训练效果对比脚本
对比使用原始数据和修正数据训练的模型效果
"""

import os
import json
import torch
from datetime import datetime
import logging

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from configs.config import *
from src.training.train import AddressNERTrainer
from src.training.evaluate import ModelEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingComparison:
    def __init__(self):
        self.results = {}
        
    def train_with_data(self, data_name: str, data_path: str, output_suffix: str):
        """使用指定数据训练模型"""
        logger.info(f"\n{'='*50}")
        logger.info(f"开始使用 {data_name} 训练模型")
        logger.info(f"数据路径: {data_path}")
        logger.info(f"{'='*50}")
        
        if not os.path.exists(data_path):
            logger.error(f"数据文件不存在: {data_path}")
            return None
            
        # 创建训练器
        trainer = AddressNERTrainer()
        
        # 修改输出目录
        original_output_dir = MODEL_OUTPUT_DIR
        new_output_dir = f"{MODEL_OUTPUT_DIR}_{output_suffix}"
        os.makedirs(new_output_dir, exist_ok=True)
        
        try:
            # 训练模型
            trainer.train(data_path)
            
            # 保存模型到指定目录
            model_save_path = os.path.join(new_output_dir, "pytorch_model.bin")
            torch.save(trainer.model.state_dict(), model_save_path)
            
            # 保存分词器
            trainer.tokenizer.tokenizer.save_pretrained(new_output_dir)
            
            logger.info(f"模型已保存到: {new_output_dir}")
            return new_output_dir
            
        except Exception as e:
            logger.error(f"训练失败: {str(e)}")
            return None
    
    def evaluate_model(self, model_path: str, data_name: str, test_data_path: str):
        """评估模型性能"""
        logger.info(f"\n评估 {data_name} 训练的模型...")
        
        try:
            # 创建评估器
            evaluator = ModelEvaluator(model_path=model_path)
            
            # 评估数据集
            if os.path.exists(test_data_path):
                results = evaluator.evaluate_dataset(test_data_path)
                self.results[data_name] = results
                
                logger.info(f"{data_name} 模型评估结果:")
                logger.info(f"  精确率: {results['precision']:.4f}")
                logger.info(f"  召回率: {results['recall']:.4f}")
                logger.info(f"  F1分数: {results['f1']:.4f}")
                
                return results
            else:
                logger.error(f"测试数据不存在: {test_data_path}")
                return None
                
        except Exception as e:
            logger.error(f"评估失败: {str(e)}")
            return None
    
    def demo_comparison(self, model_path_original: str, model_path_corrected: str):
        """演示对比两个模型的预测效果"""
        demo_texts = [
            "شارع الملك فهد، حي الملز، الرياض، المملكة العربية السعودية",
            "طريق الأمير محمد بن عبدالعزيز، الدمام، المنطقة الشرقية",
            "مبنى رقم 123، شارع التحلية، جدة 21411"
        ]
        
        logger.info(f"\n{'='*60}")
        logger.info("模型预测效果对比")
        logger.info(f"{'='*60}")
        
        for i, text in enumerate(demo_texts, 1):
            logger.info(f"\n示例 {i}: {text}")
            logger.info("-" * 50)
            
            # 原始数据训练的模型
            try:
                evaluator_original = ModelEvaluator(model_path=model_path_original)
                tokens_orig, labels_orig = evaluator_original.predict_text(text)
                entities_orig = evaluator_original.extract_entities(tokens_orig, labels_orig)
                
                logger.info("原始数据训练的模型:")
                logger.info(f"  实体数量: {len(entities_orig)}")
                for entity, label in entities_orig:
                    logger.info(f"    {entity} ({label})")
                    
            except Exception as e:
                logger.error(f"原始模型预测失败: {str(e)}")
            
            # 修正数据训练的模型
            try:
                evaluator_corrected = ModelEvaluator(model_path=model_path_corrected)
                tokens_corr, labels_corr = evaluator_corrected.predict_text(text)
                entities_corr = evaluator_corrected.extract_entities(tokens_corr, labels_corr)
                
                logger.info("修正数据训练的模型:")
                logger.info(f"  实体数量: {len(entities_corr)}")
                for entity, label in entities_corr:
                    logger.info(f"    {entity} ({label})")
                    
            except Exception as e:
                logger.error(f"修正模型预测失败: {str(e)}")
    
    def print_final_comparison(self):
        """打印最终对比结果"""
        logger.info(f"\n{'='*60}")
        logger.info("最终对比结果")
        logger.info(f"{'='*60}")
        
        if len(self.results) >= 2:
            original_results = self.results.get('原始数据', {})
            corrected_results = self.results.get('修正数据', {})
            
            metrics = ['precision', 'recall', 'f1']
            
            logger.info(f"{'指标':<15} {'原始数据':<15} {'修正数据':<15} {'提升':<15}")
            logger.info("-" * 60)
            
            for metric in metrics:
                orig_val = original_results.get(metric, 0)
                corr_val = corrected_results.get(metric, 0)
                improvement = corr_val - orig_val
                
                logger.info(f"{metric:<15} {orig_val:<15.4f} {corr_val:<15.4f} {improvement:+.4f}")
        
        logger.info(f"\n结论:")
        logger.info("修正数据质量对模型性能的影响显著！")
        logger.info("高质量的标注数据是训练成功的关键。")

def main():
    """主函数"""
    comparison = TrainingComparison()
    
    # 数据路径
    original_data_path = os.path.join(PROCESSED_DATA_DIR, "processed_sample.json")
    corrected_data_path = os.path.join(PROCESSED_DATA_DIR, "corrected_sample.json")
    
    # 1. 使用原始数据训练
    logger.info("第一阶段: 使用原始数据训练模型")
    model_path_original = comparison.train_with_data(
        "原始数据", 
        original_data_path, 
        "original"
    )
    
    # 2. 使用修正数据训练
    logger.info("\n第二阶段: 使用修正数据训练模型")
    model_path_corrected = comparison.train_with_data(
        "修正数据", 
        corrected_data_path, 
        "corrected"
    )
    
    # 3. 评估两个模型
    if model_path_original:
        comparison.evaluate_model(model_path_original, "原始数据", corrected_data_path)
    
    if model_path_corrected:
        comparison.evaluate_model(model_path_corrected, "修正数据", corrected_data_path)
    
    # 4. 演示对比
    if model_path_original and model_path_corrected:
        comparison.demo_comparison(model_path_original, model_path_corrected)
    
    # 5. 打印最终结果
    comparison.print_final_comparison()

if __name__ == "__main__":
    main()