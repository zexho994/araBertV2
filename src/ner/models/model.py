"""NER 模型定义

实现基础 NER 模型类以及基于 BERT 的具体实现。

- 重要：预测与对齐依赖 fast 分词器以提供 word_ids 信息；slow 分词器可能不支持 word_ids。
- 提示：推理阶段对置信度的阈值化仅作简单筛选，具体阈值应依据校准结果调整。
"""

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoModel, AutoConfig, BertModel,
    PreTrainedModel, PretrainedConfig
)
from typing import Dict, Any, List, Optional

class NERModelConfig(PretrainedConfig):
    """NER 模型配置类
    
    职责：
    - 封装与 NER 相关的配置（标签映射、dropout 等）
    - 兼容 transformers 的配置接口
    
    注意：
    - label2id 与 id2label 的键类型（int 或 str）在下游使用时可能混用，应尽量统一。
    """
    
    def __init__(
        self,
        num_labels: int = 2,
        hidden_dropout_prob: float = 0.1,
        classifier_dropout: Optional[float] = None,
        label2id: Optional[Dict[str, int]] = None,
        id2label: Optional[Dict[int, str]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.num_labels = num_labels
        self.hidden_dropout_prob = hidden_dropout_prob
        self.classifier_dropout = classifier_dropout
        
        if label2id is not None and id2label is not None:
            if len(label2id) != len(id2label):
                raise ValueError("label2id and id2label must have the same length")
            self.label2id = label2id
            self.id2label = id2label
        elif label2id is not None:
            self.label2id = label2id
            self.id2label = {v: k for k, v in label2id.items()}
        elif id2label is not None:
            self.id2label = id2label
            self.label2id = {v: k for k, v in id2label.items()}
        else:
            self.label2id = {}
            self.id2label = {}
        
        # TODO：统一 id2label/label2id 的键类型（全部 int 或全部 str），避免推理阶段取值时的类型不一致问题

class NERModel(PreTrainedModel):
    """NER 模型基类
    
    提供统一的 forward 接口约定与常用的预测流程封装。
    """
    
    config_class = NERModelConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """前向传播（需在子类中实现）"""
        raise NotImplementedError("Subclasses must implement forward method")
    
    def predict(
        self, 
        text: str, 
        tokenizer=None, 
        confidence_threshold: float = 0.5,
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """对输入文本进行预测（基于空白切词）"""
        if text is None:
            raise ValueError("text is required")
        words = text.split()
        return self.predict_tokens(words, tokenizer=tokenizer, confidence_threshold=confidence_threshold, device=device)

    def predict_tokens(
        self,
        words: List[str],
        tokenizer=None,
        confidence_threshold: float = 0.5,
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """对输入的词序列进行预测（避免二次切词导致的对齐偏差）
        
        参数：
            words: 词序列（与标注对齐）
            tokenizer: fast 分词器
            confidence_threshold: 概率阈值（低于则置为 'O'）
            device: 推理设备
        """
        if tokenizer is None:
            raise ValueError("Tokenizer is required for prediction")
        if device is None:
            device = next(self.parameters()).device
        
        self.eval()
        
        tokenized = tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        try:
            word_ids = tokenized.word_ids(batch_index=0)
        except TypeError:
            word_ids = tokenized.word_ids()
        
        tokenized = {k: v.to(device) for k, v in tokenized.items()}
        
        with torch.no_grad():
            outputs = self(**tokenized)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
        
        word_predictions: List[str] = []
        word_confidences: List[float] = []
        previous_word_idx = None
        for i, word_idx in enumerate(word_ids):
            if word_idx is not None and word_idx != previous_word_idx:
                if word_idx < len(words):
                    pred_id = int(predictions[0][i].item())
                    confidence = float(probabilities[0][i][pred_id].item())
                    if confidence >= confidence_threshold:
                        label = self.config.id2label.get(str(pred_id), self.config.id2label.get(pred_id, 'O'))
                    else:
                        label = 'O'
                    word_predictions.append(label)
                    word_confidences.append(confidence)
                previous_word_idx = word_idx
        
        entities = self._extract_entities(words, word_predictions, word_confidences)
        return {
            'text': ' '.join(words),
            'tokens': words,
            'labels': word_predictions,
            'confidences': word_confidences,
            'entities': entities
        }
    
    def _extract_entities(self, tokens: List[str], labels: List[str], 
                         confidences: List[float]) -> List[Dict[str, Any]]:
        """从 BIO 序列中抽取实体
        
        注意：当前对实体置信度采取两两平均的简单聚合方式，并非严格均值；可按 token 数进行加权平均。
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
                    # TODO：使用累计求和并在末尾做平均；当前做法是滚动二分平均，存在偏差
                    current_entity['confidence'] = (
                        current_entity['confidence'] + confidence
                    ) / 2
                else:
                    # 实体类型不一致，结束当前实体并开启新实体
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

class BertNERModel(NERModel):
    """基于 BERT 的 NER 模型实现
    
    由 BERT 提供上下文表示，在其顶端接线性分类器进行序列标注。
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        # 确保标签数已配置
        self.num_labels = config.num_labels
        
        # 加载 BERT 主干
        if hasattr(config, 'model_name'):
            # TODO：类名为 BertNERModel，但此处允许任何 AutoModel；可校验是否为 BERT 兼容架构
            self.bert = AutoModel.from_pretrained(config.model_name)
        else:
            self.bert = BertModel(config)
        
        # Dropout（优先使用 classifier_dropout，否则回退到 hidden_dropout_prob）
        classifier_dropout = (
            config.classifier_dropout 
            if config.classifier_dropout is not None 
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        
        # 线性分类头
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        # 初始化权重（遵循 transformers 约定）
        self.init_weights()
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        num_labels: int,
        dropout: float = 0.1,
        **kwargs
    ):
        """加载预训练 BERT 并构建用于 NER 的模型实例
        
        参数：
            pretrained_model_name_or_path: 预训练模型名或路径
            num_labels: 标签数量
            dropout: 分类头 dropout
            **kwargs: 其他配置项（透传到 NERModelConfig）
        
        返回：
            BertNERModel 实例
        """
        # 读取基础配置
        base_config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        
        # 构造 NER 配置
        config = NERModelConfig(
            vocab_size=base_config.vocab_size,
            hidden_size=base_config.hidden_size,
            num_hidden_layers=base_config.num_hidden_layers,
            num_attention_heads=base_config.num_attention_heads,
            intermediate_size=base_config.intermediate_size,
            hidden_act=base_config.hidden_act,
            hidden_dropout_prob=base_config.hidden_dropout_prob,
            attention_probs_dropout_prob=base_config.attention_probs_dropout_prob,
            max_position_embeddings=base_config.max_position_embeddings,
            type_vocab_size=base_config.type_vocab_size,
            initializer_range=base_config.initializer_range,
            layer_norm_eps=base_config.layer_norm_eps,
            pad_token_id=base_config.pad_token_id,
            position_embedding_type=getattr(base_config, 'position_embedding_type', 'absolute'),
            num_labels=num_labels,
            classifier_dropout=dropout,
            model_name=pretrained_model_name_or_path,
            **kwargs
        )

        print(f"=> bert ner config: {config}")
        
        # 再次确保标签数
        config.num_labels = num_labels
        
        # 创建模型
        model = cls(config)
        
        # 载入预训练主干权重
        # TODO: 这里可以优化，将加载权重的操作提取到类外，避免重复加载
        model.bert = AutoModel.from_pretrained(pretrained_model_name_or_path)
        
        return model
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """前向传播
        
        参数：
            input_ids: Token ID 序列
            attention_mask: 注意力掩码
            token_type_ids: 句子类型 ID（对 BERT 有效）
            labels: 训练标签（可选）
        
        返回：
            dict：包含 loss（可为 None）、logits、hidden_states、attentions
        """
        # BERT 前向
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # 取序列输出
        sequence_output = outputs.last_hidden_state
        
        # Dropout
        sequence_output = self.dropout(sequence_output)
        
        # 分类
        logits = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # 仅在有效位置计算损失
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
        
        # TODO：
        # - 可选引入 CRF 层提升序列一致性
        # - 支持输出中间层特征以便蒸馏/分析
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            'attentions': outputs.attentions if hasattr(outputs, 'attentions') else None
        }
    
    def get_input_embeddings(self):
        """获取输入词向量层"""
        return self.bert.embeddings.word_embeddings
    
    def set_input_embeddings(self, value):
        """设置输入词向量层"""
        self.bert.embeddings.word_embeddings = value
    
    def resize_token_embeddings(self, new_num_tokens: int):
        """调整词表大小（重建并部分拷贝旧权重）
        
        注意：此实现未调用 transformers 的通用接口，可能遗漏一些权重绑定逻辑。
        """
        old_embeddings = self.get_input_embeddings()
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.set_input_embeddings(new_embeddings)
        return self.get_input_embeddings()
    
    def _get_resized_embeddings(
        self, old_embeddings: nn.Embedding, new_num_tokens: int
    ) -> nn.Embedding:
        """内部：创建新词向量表并拷贝前 min(N_old, N_new) 行权重"""
        old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
        
        if old_num_tokens == new_num_tokens:
            return old_embeddings
        
        # 创建新嵌入
        new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim)
        new_embeddings.to(old_embeddings.weight.device, dtype=old_embeddings.weight.dtype)
        
        # 复制旧权重
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        new_embeddings.weight.data[:num_tokens_to_copy, :] = old_embeddings.weight.data[:num_tokens_to_copy, :]
        
        # TODO：当扩大词表时，新增行可考虑用旧权重统计分布进行初始化
        return new_embeddings
    
    def freeze_bert_layers(self, num_layers: int = 0):
        """冻结 BERT 层参数
        
        参数：
            num_layers: 冻结层数（0 表示冻结全部）
        """
        if num_layers == 0:
            # 冻结全部 BERT 参数
            for param in self.bert.parameters():
                param.requires_grad = False
        else:
            # 冻结指定数量的底层编码层
            layers_to_freeze = self.bert.encoder.layer[:num_layers]
            for layer in layers_to_freeze:
                for param in layer.parameters():
                    param.requires_grad = False
        # TODO：支持按层名/范围更灵活地冻结
    
    def unfreeze_bert_layers(self):
        """解冻全部 BERT 参数"""
        for param in self.bert.parameters():
            param.requires_grad = True
    
    def get_model_size(self) -> Dict[str, int]:
        """获取模型规模信息（参数总数/可训练数/冻结数）"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': total_params - trainable_params
        }
    
    def save_pretrained(self, save_directory: str):
        """保存模型权重与配置
        
        注意：此处仅保存模型与配置，不包含 tokenizer；需要时请单独保存 tokenizer。
        """
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        # 保存模型权重
        model_path = os.path.join(save_directory, 'pytorch_model.bin')
        torch.save(self.state_dict(), model_path)
        
        # 保存配置
        config_path = os.path.join(save_directory, 'config.json')
        # TODO：变量 config_path 未使用，可移除或用于手动写入配置文件路径
        self.config.save_pretrained(save_directory)