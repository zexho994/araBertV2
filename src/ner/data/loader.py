"""NER 数据加载器

用于 NER 任务的 PyTorch Dataset 与 DataLoader 实现。

- 重要：采用子词对齐策略（word-level → subword-level），仅词的首个子词继承词级标签；其余位置使用 pad_token_label_id（默认 -100），从而在损失与评估中被忽略。
- 调试：设置环境变量 DEBUG_ALIGNMENT=1 可输出一次性对齐表，帮助核对分词与标签对齐是否正确。
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional, Tuple
from transformers import AutoTokenizer
import os
from ..utils import NERLogger

class NERDataset(Dataset):
    """用于 NER 的 PyTorch Dataset

    职责：
    - 接收词序列与词级标签
    - 使用分词器进行子词化，并将词级标签精确对齐到子词级
    - 产出模型训练/评估直接可用的 input_ids、attention_mask 与 labels
    """
    
    def __init__(self, dataset: List[Dict[str, Any]], tokenizer, 
                 label2id: Dict[str, int], max_length: int = 512,
                 pad_token_label_id: int = -100):
        """初始化 NER Dataset
        
        参数：
            dataset: 样本列表，元素包含 'tokens' 与 'labels'
            tokenizer: 分词器实例（需为 fast tokenizer 以支持 word_ids 对齐）
            label2id: 标签到 ID 的映射
            max_length: 最大序列长度（会进行截断与填充）
            pad_token_label_id: 用于填充/忽略位置的标签 ID（通常为 -100）
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.pad_token_label_id = pad_token_label_id
        
        # 重要：将原始样本预处理为模型可直接消费的张量形式
        # TODO：可考虑在大量数据时引入缓存/懒加载机制以降低启动开销
        self.processed_datasets = self._process_datasets()
    
    def _process_datasets(self) -> List[Dict[str, Any]]:
        """将原始数据集转为模型输入格式"""
        processed = []
        
        for data in self.dataset:
            tokens = data['tokens'] # 词列表
            labels = data['labels'] # 标签列表
            
            # 分词并对齐标签
            tokenized = self._tokenize_and_align_labels(tokens, labels)
            
            if tokenized:
                processed.append(tokenized)
        
        return processed
    
    def _tokenize_and_align_labels(self, tokens: List[str], labels: List[str]) -> Optional[Dict[str, Any]]:
        """对输入词列表进行分词，并将词级标签对齐到子词级。
        策略：
          - 每个词仅其首个子词继承该词的标签
          - 该词其余子词位置使用 pad_token_label_id（如 -100），在损失/指标中被忽略
          - 特殊符号（CLS/SEP/PAD）位置同样使用 pad_token_label_id
        """
        # 将词序列作为已分好词的输入传给分词器
        tokenized_inputs = self.tokenizer(
            tokens, # 词列表
            is_split_into_words=True, # 输入已分词，无需再切词
            max_length=self.max_length, # 最大长度
            padding='max_length', # 填充
            truncation=True, # 截断
            return_tensors='pt' # 转换为 PyTorch 张量
        )

        # 兼容不同 transformers 版本对 BatchEncoding.word_ids 的签名差异
        try:
            word_ids = tokenized_inputs.word_ids(batch_index=0)
        except TypeError:
            word_ids = tokenized_inputs.word_ids()

        # 将标签按照分词后的子词对齐
        aligned_labels: List[int] = []
        previous_word_idx: Optional[int] = None

        for word_idx in word_ids:
            if word_idx is None:
                # 特殊符号（CLS, SEP, PAD）
                aligned_labels.append(self.pad_token_label_id)
            elif word_idx != previous_word_idx:
                # 该词的首个子词 → 赋予对应词级标签
                if word_idx < len(labels):
                    label = labels[word_idx]
                    aligned_labels.append(self.label2id.get(label, self.pad_token_label_id))
                else:
                    # ERROR：此分支表明 word_ids 的索引超出 labels 长度，通常意味着上游 tokens/labels 未对齐
                    # TODO：可在此处记录告警日志或直接丢弃样本
                    aligned_labels.append(self.pad_token_label_id)
            else:
                # 该词的后续子词 → 忽略
                aligned_labels.append(self.pad_token_label_id)

            previous_word_idx = word_idx

        # 可选：一次性打印对齐调试信息（仅当设置 DEBUG_ALIGNMENT=1）
        # if os.environ.get("DEBUG_ALIGNMENT", "0") == "1" and not getattr(self, "_alignment_debug_printed", False):
        #     try:
        #         self._print_alignment_debug(tokenized_inputs, tokens, labels, aligned_labels)
        #     except Exception:
        #         pass
        #     self._alignment_debug_printed = True

        return {
            'input_ids': tokenized_inputs['input_ids'].squeeze(),
            'attention_mask': tokenized_inputs['attention_mask'].squeeze(),
            'labels': torch.tensor(aligned_labels, dtype=torch.long),
            'original_tokens': tokens,
            'original_labels': labels
        }

    def _print_alignment_debug(
        self,
        tokenized_inputs: Any,
        tokens: List[str],
        labels: List[str],
        aligned_labels: List[int]
    ) -> None:
        """当 DEBUG_ALIGNMENT=1 时打印对齐表，便于快速人工核验。"""
        # 兼容不同 transformers 版本对 BatchEncoding.word_ids 的签名差异
        try:
            word_ids = tokenized_inputs.word_ids(batch_index=0)
        except TypeError:
            word_ids = tokenized_inputs.word_ids()
        input_ids = tokenized_inputs['input_ids'][0]
        sub_tokens = self.tokenizer.convert_ids_to_tokens(input_ids.tolist())

        id2label = {v: k for k, v in self.label2id.items()}
        rows = []
        rows.append("==== DEBUG ALIGNMENT (train-time) ====")
        rows.append("Words and labels:")
        rows.append(" ".join([f"{w}({y})" for w, y in zip(tokens, labels)]))
        rows.append("Sub-tokens / word_id / aligned_label:")
        for idx, (tok, wid, lab_id) in enumerate(zip(sub_tokens, word_ids, aligned_labels)):
            lab = id2label.get(int(lab_id), "IGN") if lab_id != self.pad_token_label_id else "IGN"
            rows.append(f"{idx:>3}: {tok:>20} | word_id={str(wid):>3} | label={lab}")
        print("\n".join(rows))
    
    def __len__(self) -> int:
        """返回样本数量"""
        return len(self.processed_datasets)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """返回单条样本的张量化结果"""
        return self.processed_datasets[idx]

class NERDataLoader:
    """NER 数据加载器管理器
    
    职责：
    - 基于给定分词器与标签映射，创建 Dataset 与 DataLoader
    - 提供训练/验证/测试三种数据加载器的便捷构建入口
    """
    
    def __init__(self, tokenizer_name: str, label2id: Dict[str, int], 
                 max_length: int = 512, pad_token_label_id: int = -100, logger: Optional[NERLogger] = None):
        """初始化 NER 数据加载器
        
        参数：
            tokenizer_name: 分词器名称或路径
            label2id: 标签到 ID 的映射
            max_length: 最大序列长度
            pad_token_label_id: 用于忽略位置的标签 ID
        """
        # 始终使用 fast 分词器以启用 word_ids 对齐
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.label2id = label2id
        self.max_length = max_length
        self.pad_token_label_id = pad_token_label_id
        self.logger = logger
        
        if self.tokenizer.pad_token is None:
            raise ValueError("Tokenizer must have a pad_token")
    
    def create_dataset_loader(self, dataset: List[Dict[str, Any]]) -> NERDataset:
        """由数据集创建 NERDataset 实例
        
        参数：
            dataset: 数据集列表
            
        返回：
            NERDataset 实例
        """
        dataset_loader = NERDataset(
            dataset=dataset,
            tokenizer=self.tokenizer,
            label2id=self.label2id,
            max_length=self.max_length,
            pad_token_label_id=self.pad_token_label_id
        )
        self.logger.info(f"Created dataset loader with {len(dataset)} samples")
        return dataset_loader
    
    def create_dataloader(self, dataset: NERDataset, batch_size: int = 16, 
                         shuffle: bool = True, num_workers: int = 0) -> DataLoader:
        """创建 PyTorch DataLoader
        
        参数：
            dataset: NER 数据集
            batch_size: 批大小
            shuffle: 是否打乱
            num_workers: DataLoader 子进程数量（Windows 下需注意多进程启动方式）
            
        返回：
            DataLoader 实例
        """
        # TODO：可考虑使用 transformers 的 DataCollatorForTokenClassification 或基于 tokenizer 的动态 padding，降低显存浪费
        dl = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate_fn
        )
        self.logger.info(f"Created dataloader with batch_size={batch_size}, shuffle={shuffle}, num_workers={num_workers}")
        return dl
    
    def _collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """自定义合批函数：将样本堆叠为批次张量"""
        # 重要：此处直接 stack，依赖于上游已按 max_length 对齐（固定长度）。若改为动态 padding，应在此处改为 pad_sequence。
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        
        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
        
        # 如果batch中包含原始tokens和labels，也传递它们（用于评估时的对齐）
        if 'original_tokens' in batch[0]:
            result['original_tokens'] = [item.get('original_tokens', []) for item in batch]
        if 'original_labels' in batch[0]:
            result['original_labels'] = [item.get('original_labels', []) for item in batch]
        
        return result
    
    def prepare_loaders(self, train_dataset: List[Dict[str, Any]], 
                           val_dataset: List[Dict[str, Any]],
                           test_dataset: Optional[List[Dict[str, Any]]] = None,
                           batch_size: int = 16, 
                           num_workers: int = 0) -> Dict[str, DataLoader]:
        """构建训练/验证/测试的数据加载器集合
        
        参数：
            train_examples: 训练样本
            val_examples: 验证样本
            test_examples: 测试样本（可选）
            batch_size: 批大小
            num_workers: DataLoader 子进程数量
            
        返回：
            包含 'train'/'val'/'test' 的 DataLoader 字典
        """
        data_loaders = {}
        
        # 构建训练集数据加载器
        train_dataset = self.create_dataset_loader(train_dataset)
        data_loaders['train'] = self.create_dataloader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        
        # 构建验证集数据加载器
        val_dataset = self.create_dataset_loader(val_dataset)
        data_loaders['val'] = self.create_dataloader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        
        if test_dataset:
            # 构建测试集数据加载器
            test_dataset = self.create_dataset_loader(test_dataset)
            data_loaders['test'] = self.create_dataloader(
                test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
            )
        
        return data_loaders
    
    def get_tokenizer(self):
        """返回分词器实例"""
        return self.tokenizer

class NERTokenizer:
    """NER 预测/交互流程的分词与解码工具封装"""
    
    def __init__(self, tokenizer_name: str, max_length: int = 512):
        """初始化 NER 分词工具
        
        参数：
            tokenizer_name: 分词器名称或路径
            max_length: 最大序列长度
        """
        # 使用 fast 分词器以支持 word_ids 对齐（用于预测/交互场景）
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.max_length = max_length
        
        # TODO：与数据加载部分一致，若无 pad_token，可考虑更稳妥的回退策略
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def tokenize_text(self, text: str) -> Dict[str, Any]:
        """对输入文本分词并返回与词级对齐相关的信息（用于预测）
        
        参数：
            text: 原始输入文本
            
        返回：
            包含 input_ids、attention_mask、word_ids、words 的字典
        """
        # 简单按空白切分为词（针对已空白分词的语言/数据）
        words = text.split()
        
        # 分词
        tokenized = self.tokenizer(
            words,
            is_split_into_words=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 兼容不同 transformers 版本对 BatchEncoding.word_ids 的签名差异
        try:
            word_ids = tokenized.word_ids(batch_index=0)
        except TypeError:
            word_ids = tokenized.word_ids()
        
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'word_ids': word_ids,
            'words': words
        }
    
    def decode_predictions(self, predictions: torch.Tensor, word_ids: List[Optional[int]], 
                         words: List[str], id2label: Dict[int, str]) -> List[Tuple[str, str]]:
        """将模型的 token 级预测解码为 (word, label) 对
        
        参数：
            predictions: 模型 token 级预测（通常是 argmax 后的 ID 序列）
            word_ids: 分词与词的对齐索引
            words: 原始词序列
            id2label: ID 到标签的映射
            
        返回：
            (word, label) 列表（仅保留每个词的首个子词位置）
        """
        # 将预测 ID 转为标签字符串
        predicted_labels = [id2label.get(pred.item(), 'O') for pred in predictions]
        
        # 按词级对齐聚合
        word_labels = []
        previous_word_idx = None
        
        for word_idx, label in zip(word_ids, predicted_labels):
            if word_idx is not None and word_idx != previous_word_idx:
                if word_idx < len(words):
                    word_labels.append((words[word_idx], label))
                previous_word_idx = word_idx
        
        # TODO：可选支持将 BIOES 等标签解码为实体片段（span）结构
        return word_labels
    
    def batch_tokenize(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """批量分词（预测批处理场景）
        
        参数：
            texts: 输入文本列表
            
        返回：
            批量分词后的张量
        """
        return self.tokenizer(
            texts,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
    
    def get_vocab_size(self) -> int:
        """返回词表大小"""
        return len(self.tokenizer)
    
    def get_special_tokens(self) -> Dict[str, str]:
        """返回特殊符号"""
        return {
            'pad_token': self.tokenizer.pad_token,
            'cls_token': self.tokenizer.cls_token,
            'sep_token': self.tokenizer.sep_token,
            'unk_token': self.tokenizer.unk_token,
            'mask_token': getattr(self.tokenizer, 'mask_token', None)
        }