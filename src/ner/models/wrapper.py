"""Transformers 兼容性模型封装

提供封装以便将标准 transformers 模型适配为与 NEREvaluator 接口兼容的形式。

- 重要：预测对齐依赖 fast 分词器（需支持 word_ids）。
- 提示：该封装适用于已按照 token 分类任务训练的模型（AutoModelForTokenClassification）。
"""

import torch
from typing import Dict, Any, List, Optional
from transformers import AutoTokenizer, AutoModelForTokenClassification


class TransformersNERModelWrapper:
    """将 transformers 标准模型封装为 NEREvaluator 可用的接口
    
    职责：
    - 统一预测接口（predict），输出 tokens/labels/confidences 与实体抽取结果
    - 管理设备迁移与 eval/train 状态切换
    """
    
    def __init__(self, model: AutoModelForTokenClassification, tokenizer: AutoTokenizer, 
                 id2label: Dict[int, str], label2id: Dict[str, int]):
        """初始化封装
        
        参数：
            model: transformers 标准 token 分类模型
            tokenizer: 分词器
            id2label: ID→标签 映射
            label2id: 标签→ID 映射
        """
        self.model = model
        self.tokenizer = tokenizer
        self.id2label = id2label
        self.label2id = label2id
        self.device = next(model.parameters()).device
        
        # 确保评估模式
        self.model.eval()
    
    def predict(
        self, 
        text: str, 
        tokenizer=None, 
        confidence_threshold: float = 0.5,
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """对输入文本进行预测
        
        参数：
            text: 输入文本
            tokenizer: 传入的分词器（忽略，使用封装内的 tokenizer）
            confidence_threshold: 置信度阈值
            device: 推理设备（忽略，使用封装内的 device）
        
        返回：
            与 NEREvaluator 兼容的预测结果
        """
        # 使用封装内的分词器与设备
        tokenizer = self.tokenizer
        device = self.device
        
        # 分词
        words = text.split()
        tokenized = tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # 获取对齐信息
        # ERROR：若 tokenizer 不是 fast 实现，word_ids 可能不可用；需保证使用 fast tokenizer
        word_ids = tokenized.word_ids(batch_index=0)
        
        # 迁移到设备
        tokenized = {k: v.to(device) for k, v in tokenized.items()}
        
        # 前向推理
        with torch.no_grad():
            outputs = self.model(**tokenized)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
        
        # 对齐到词级（仅保留每个词的首个子词）
        word_predictions = []
        word_confidences = []
        previous_word_idx = None
        
        for i, word_idx in enumerate(word_ids):
            if word_idx is not None and word_idx != previous_word_idx:
                if word_idx < len(words):
                    pred_id = predictions[0][i].item()
                    confidence = probabilities[0][i][pred_id].item()
                    
                    if confidence >= confidence_threshold:
                        label = self.id2label.get(pred_id, 'O')
                    else:
                        label = 'O'
                    
                    word_predictions.append(label)
                    word_confidences.append(confidence)
                
                previous_word_idx = word_idx
        
        # 实体抽取（基于 BIO）
        entities = self._extract_entities(words, word_predictions, word_confidences)
        
        return {
            'text': text,
            'tokens': words,
            'labels': word_predictions,
            'confidences': word_confidences,
            'entities': entities
        }
    
    def _extract_entities(self, tokens: List[str], labels: List[str], 
                         confidences: List[float]) -> List[Dict[str, Any]]:
        """从 BIO 序列抽取实体
        
        注意：当前采用滚动二分平均更新实体置信度，建议替换为累计求平均或加权策略。
        """
        entities = []
        current_entity = None
        
        for i, (token, label, confidence) in enumerate(zip(tokens, labels, confidences)):
            if label.startswith('B-'):
                # 新实体开始
                if current_entity:
                    entities.append(current_entity)
                
                entity_type = label[2:]
                current_entity = {
                    'type': entity_type,
                    'tokens': [token],
                    'start': i,
                    'end': i + 1,
                    'text': token,
                    'confidence': confidence
                }
            
            elif label.startswith('I-') and current_entity:
                # 实体延续
                entity_type = label[2:]
                if current_entity['type'] == entity_type:
                    current_entity['tokens'].append(token)
                    current_entity['end'] = i + 1
                    current_entity['text'] += ' ' + token
                    # TODO：使用累计求和并在末尾求平均，减少序依赖偏差
                    current_entity['confidence'] = (
                        current_entity['confidence'] + confidence
                    ) / 2
                else:
                    # 实体类型不一致，结束当前并新开实体
                    entities.append(current_entity)
                    current_entity = {
                        'type': entity_type,
                        'tokens': [token],
                        'start': i,
                        'end': i + 1,
                        'text': token,
                        'confidence': confidence
                    }
            
            else:
                # O 或实体结束
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        # 收尾
        if current_entity:
            entities.append(current_entity)
        
        return entities
    
    def to(self, device):
        """将模型迁移到设备"""
        self.model.to(device)
        self.device = device
        return self
    
    def eval(self):
        """切换评估模式"""
        self.model.eval()
        return self
    
    def train(self, mode=True):
        """切换训练模式"""
        self.model.train(mode)
        return self
    
    @property
    def config(self):
        """访问底层模型配置"""
        return self.model.config
    
    def __call__(self, *args, **kwargs):
        """转发调用到底层模型"""
        return self.model(*args, **kwargs)


def wrap_transformers_model(model_path: str) -> TransformersNERModelWrapper:
    """加载并封装标准 transformers 模型
    
    参数：
        model_path: 已保存模型的路径
    
    返回：
        与 NEREvaluator 兼容的封装模型
    """
    # 加载模型与分词器
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 获取标签映射
    id2label = model.config.id2label
    label2id = model.config.label2id
    
    # TODO：确保 tokenizer 为 fast 类型；否则 word_ids 可能不可用
    
    # 创建封装
    wrapper = TransformersNERModelWrapper(model, tokenizer, id2label, label2id)
    
    return wrapper


def create_compatible_evaluator(model_path: str, device=None):
    """用封装模型创建 NEREvaluator
    
    参数：
        model_path: 模型路径
        device: 推理设备
    
    返回：
        带封装模型的 NEREvaluator 实例
    """
    from ..evaluation.evaluator import NEREvaluator
    
    # 封装模型
    wrapped_model = wrap_transformers_model(model_path)
    
    # 迁移设备
    if device is not None:
        wrapped_model.to(device)
    
    # 标签列表
    # ERROR：若 id2label 的键非连续整数（或为字符串），按 range(len(...)) 可能取错；建议根据键排序/类型转换
    label_list = [wrapped_model.id2label[i] for i in range(len(wrapped_model.id2label))]
    
    # 创建 evaluator
    evaluator = NEREvaluator(
        model=wrapped_model,
        tokenizer=wrapped_model.tokenizer,
        label_list=label_list,
        device=wrapped_model.device
    )
    
    return evaluator