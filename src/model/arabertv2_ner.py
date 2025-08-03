"""
AraBERTv2 NER模型定义
用于阿拉伯语地址解析的命名实体识别模型
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import Optional, Tuple, Dict, Any


class AraBERTv2NER(nn.Module):
    """基于AraBERTv2的命名实体识别模型"""
    
    def __init__(self, model_name: str, num_labels: int, dropout_rate: float = 0.1):
        super(AraBERTv2NER, self).__init__()
        
        self.num_labels = num_labels
        self.model_name = model_name
        
        # 加载AraBERTv2模型
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name, config=self.config)
        
        # 分类层
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化分类层权重"""
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            input_ids: 输入token ids
            attention_mask: 注意力掩码
            token_type_ids: token类型ids
            labels: 标签（训练时使用）
            
        Returns:
            包含logits和loss的字典
        """
        # BERT编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        
        # 获取序列输出
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        
        # 分类
        logits = self.classifier(sequence_output)
        
        result = {"logits": logits}
        
        # 计算损失（如果提供了标签）
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            # 只计算有效token的损失
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss,
                    labels.view(-1),
                    torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
            result["loss"] = loss
        
        return result
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        预测模式
        
        Args:
            input_ids: 输入token ids
            attention_mask: 注意力掩码
            token_type_ids: token类型ids
            
        Returns:
            预测的标签
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            predictions = torch.argmax(outputs["logits"], dim=-1)
        
        return predictions


class AraBERTv2Tokenizer:
    """AraBERTv2分词器包装类"""
    
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model_name = model_name
    
    def tokenize_and_align_labels(
        self,
        texts: list,
        labels: list,
        max_length: int = 512
    ) -> Dict[str, torch.Tensor]:
        """
        分词并对齐标签
        
        Args:
            texts: 文本列表
            labels: 标签列表
            max_length: 最大长度
            
        Returns:
            分词后的结果
        """
        tokenized_inputs = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
            is_split_into_words=True
        )
        
        aligned_labels = []
        for i, label in enumerate(labels):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            aligned_label = []
            previous_word_idx = None
            
            for word_idx in word_ids:
                if word_idx is None:
                    aligned_label.append(-100)
                elif word_idx != previous_word_idx:
                    aligned_label.append(label[word_idx])
                else:
                    aligned_label.append(-100)
                previous_word_idx = word_idx
            
            aligned_labels.append(aligned_label)
        
        tokenized_inputs["labels"] = torch.tensor(aligned_labels)
        return tokenized_inputs