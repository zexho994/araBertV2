"""
辅助工具函数
"""

import os
import json
import torch
import random
import numpy as np
from typing import List, Dict, Any
import logging


def set_seed(seed: int = 42):
    """
    设置随机种子以确保结果可复现
    
    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # 确保CUDA操作的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_file: str = None, level: int = logging.INFO):
    """
    设置日志配置
    
    Args:
        log_file: 日志文件路径
        level: 日志级别
    """
    handlers = [logging.StreamHandler()]
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def load_json(file_path: str) -> Any:
    """
    加载JSON文件
    
    Args:
        file_path: 文件路径
        
    Returns:
        JSON数据
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, file_path: str):
    """
    保存数据为JSON文件
    
    Args:
        data: 要保存的数据
        file_path: 文件路径
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def count_parameters(model: torch.nn.Module) -> int:
    """
    计算模型参数数量
    
    Args:
        model: PyTorch模型
        
    Returns:
        参数数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """
    格式化时间显示
    
    Args:
        seconds: 秒数
        
    Returns:
        格式化的时间字符串
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def create_directories(paths: List[str]):
    """
    创建目录列表
    
    Args:
        paths: 目录路径列表
    """
    for path in paths:
        os.makedirs(path, exist_ok=True)


def get_device_info() -> Dict[str, Any]:
    """
    获取设备信息
    
    Returns:
        设备信息字典
    """
    device_info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
        "device_name": torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU"
    }
    
    return device_info


def print_model_info(model: torch.nn.Module):
    """
    打印模型信息
    
    Args:
        model: PyTorch模型
    """
    param_count = count_parameters(model)
    device_info = get_device_info()
    
    print("=== 模型信息 ===")
    print(f"可训练参数数量: {param_count:,}")
    print(f"模型大小: {param_count * 4 / 1024 / 1024:.2f} MB (假设float32)")
    
    print("\\n=== 设备信息 ===")
    print(f"CUDA可用: {device_info['cuda_available']}")
    if device_info['cuda_available']:
        print(f"CUDA设备数量: {device_info['cuda_device_count']}")
        print(f"当前设备: {device_info['current_device']}")
        print(f"设备名称: {device_info['device_name']}")
    else:
        print("使用CPU")


def validate_data_format(data: List[Dict[str, Any]]) -> bool:
    """
    验证数据格式是否正确
    
    Args:
        data: 数据列表
        
    Returns:
        是否格式正确
    """
    required_keys = ["tokens", "labels", "label_ids"]
    
    for i, item in enumerate(data):
        # 检查必需的键
        for key in required_keys:
            if key not in item:
                print(f"样本 {i} 缺少键: {key}")
                return False
        
        # 检查长度一致性
        if len(item["tokens"]) != len(item["labels"]) or len(item["tokens"]) != len(item["label_ids"]):
            print(f"样本 {i} 的tokens、labels和label_ids长度不一致")
            return False
    
    return True


def split_data(data: List[Any], train_ratio: float = 0.8, val_ratio: float = 0.1) -> tuple:
    """
    分割数据为训练集、验证集和测试集
    
    Args:
        data: 数据列表
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        
    Returns:
        (train_data, val_data, test_data) 元组
    """
    total_size = len(data)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    
    # 随机打乱数据
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    
    train_data = shuffled_data[:train_size]
    val_data = shuffled_data[train_size:train_size + val_size]
    test_data = shuffled_data[train_size + val_size:]
    
    return train_data, val_data, test_data


def calculate_class_weights(labels: List[List[int]], num_classes: int) -> torch.Tensor:
    """
    计算类别权重以处理不平衡数据
    
    Args:
        labels: 标签列表
        num_classes: 类别数量
        
    Returns:
        类别权重张量
    """
    # 统计每个类别的频次
    class_counts = [0] * num_classes
    total_count = 0
    
    for label_seq in labels:
        for label in label_seq:
            if 0 <= label < num_classes:
                class_counts[label] += 1
                total_count += 1
    
    # 计算权重
    weights = []
    for count in class_counts:
        if count > 0:
            weight = total_count / (num_classes * count)
        else:
            weight = 1.0
        weights.append(weight)
    
    return torch.tensor(weights, dtype=torch.float32)