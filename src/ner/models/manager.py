"""NER 模型管理器

负责模型的加载、保存、版本管理与元信息登记。

- 重要：本管理器假定模型目录包含模型权重与配置（参考 save_model 的保存布局）。
- 提示：若直接加载标准 transformers 目录（非本工具保存格式），需确保存在 label2id/id2label 信息。
"""

import json
import torch
import shutil
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import hashlib

from .model import NERModel, BertNERModel
from ..utils import NERLogger

class NERModelManager:
    """NER 模型管理器
    
    职责：
    - 统一管理模型的存储位置、元信息注册（registry）、加载/保存/备份/恢复
    - 便于基于国家/类型进行筛选与检索
    """
    
    def __init__(self, model_dir: str = "data/ner/models", logger: Optional[NERLogger] = None):
        """初始化管理器
        
        参数：
            model_dir: 模型存储的根目录
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # 复用外部传入的 logger，否则就地创建
        self.logger = logger if logger is not None else NERLogger(name="model_manager", log_dir=self.model_dir)
        
        # 模型注册表文件
        self.registry_file = self.model_dir / "model_registry.json"
        self.registry = self._load_registry()  # eg. 'data/ner/models/model_registry.json'
    
    def _load_registry(self) -> Dict[str, Any]:
        """加载模型注册表
        
        注意：注册表损坏或 JSON 解析失败时将回退为空结构；可考虑增加备份策略。
        """
        if self.registry_file.exists():
            with open(self.registry_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {'models': {}}
    
    def _save_registry(self):
        """保存模型注册表"""
        with open(self.registry_file, 'w', encoding='utf-8') as f:
            json.dump(self.registry, f, indent=2, ensure_ascii=False)
    
    def _generate_model_id(self, country: str, model_type: str, timestamp: str) -> str:
        """生成模型 ID（短 MD5 截断）"""
        base_string = f"{country}_{model_type}_{timestamp}"
        return hashlib.md5(base_string.encode()).hexdigest()[:8]
    
    def register_model(self, model_path: str, country: str, model_type: str = "bert",
                      description: str = "", metadata: Optional[Dict[str, Any]] = None) -> str:
        """注册新模型到 registry
        
        参数：
            model_path: 模型目录路径
            country: 国家/区域代码
            model_type: 模型类型（当前支持 'bert'）
            description: 模型描述
            metadata: 额外元数据
        
        返回：
            model_id
        """
        timestamp = datetime.now().isoformat()
        model_id = self._generate_model_id(country, model_type, timestamp)
        
        # 创建注册条目
        model_entry = {
            'id': model_id,
            'country': country,
            'type': model_type,
            'path': str(model_path),
            'description': description,
            'created_at': timestamp,
            'metadata': metadata or {},
            'status': 'active'
        }
        
        # 写入注册表
        self.registry['models'][model_id] = model_entry
        self._save_registry()
        
        self.logger.info(f"Registered model {model_id} for {country}")
        return model_id
    
    def load_model(self, model_identifier: str) -> NERModel:
        """
        加载模型
        """

        model_path = Path(model_identifier)

        # 1. 检查模型是否已登记并更新访问时间
        model_id = self._find_model_id_by_path(model_path)
        if model_id:
            # 更新访问时间
            if 'last_accessed' not in self.registry['models'][model_id]:
                self.registry['models'][model_id]['last_accessed'] = datetime.now().isoformat()
            else:
                self.registry['models'][model_id]['last_accessed'] = datetime.now().isoformat()
            self._save_registry()
            self.logger.info(f"Model {model_id} found in registry, updated access time")
        else:
            self.logger.warning(f"Model {model_identifier} not found in registry, loading anyway")

        # 读取配置
        config_path = model_path / "config.json"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                label2id = config_data.get('label2id', {})
                id2label = config_data.get('id2label', {})
                lora_enabled = config_data.get('lora_enabled', False)
        else:
            raise FileNotFoundError(f"Model configuration not found: {config_path}")
        
        if lora_enabled:
            return self._load_lora_ultra_optimized(model_path, label2id, id2label)
        else:
            return self._load_full_ultra_optimized(model_path, label2id, id2label)

    def _load_lora_ultra_optimized(self, model_path: Path, label2id: dict, id2label: dict) -> NERModel:
        """超优化LoRA加载：只加载一次"""
        try:
            from peft import PeftModel
            
            # 直接从LoRA路径加载，PEFT会自动处理基础模型
            model = BertNERModel.from_pretrained(
                str(model_path),
                num_labels=len(label2id),
                id2label=id2label,
                label2id=label2id
            )
            lora_model = PeftModel.from_pretrained(
                model,
                model_path,
                is_trainable=True
            )
            self.logger.info(f"Successfully loaded LoRA model from {model_path}")
            return lora_model
            
        except ImportError as exc:
            self.logger.error("PEFT library not available. Cannot load LoRA model.")
            raise ImportError("PEFT library is required for LoRA model loading. Install with: pip install peft") from exc
        except Exception as e:
            self.logger.error(f"Failed to load LoRA model: {e}")
            raise e

    def _load_full_ultra_optimized(self, model_path: Path, label2id: dict, id2label: dict) -> NERModel:
        """超优化完整模型加载：只加载一次"""
        # 直接使用 from_pretrained，它会自动加载权重
        model = BertNERModel.from_pretrained(
            str(model_path),
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id
        )
        self.logger.info(f"Successfully loaded full model from {model_path}")
        return model

    def _find_model_id_by_path(self, model_path: Path) -> Optional[str]:
        """通过路径查找模型ID"""
        # 在注册表中查找匹配的模型路径
        for model_id, model_info in self.registry['models'].items():
            if model_info.get('path') == str(model_path):
                return model_id
        
        # 如果直接路径匹配失败，尝试相对路径匹配
        try:
            model_path_resolved = model_path.resolve()
            for model_id, model_info in self.registry['models'].items():
                if Path(model_info.get('path', '')).resolve() == model_path_resolved:
                    return model_id
        except Exception:
            pass
        
        return None

    def _detect_model_type(self, model_path: str) -> str:
        """从路径推断模型类型
        
        策略：
        1. 检查 config.json 中的 lora_enabled 标志
        2. 检查是否存在 adapter_config.json（LoRA 特有文件）
        3. 存在 'pytorch_model.bin' 或 'model.safetensors' 视为 BERT 模型
        
        # TODO：当前逻辑较为简化，后续可根据配置或文件结构更准确地区分不同模型类型。
        """
        model_path = Path(model_path)
        
        # 首先检查 config.json 中的 LoRA 标志
        config_path = model_path / "config.json"
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                    if config_data.get('lora_enabled', False):
                        return "bert"  # LoRA 模型仍然是 BERT 类型，但需要特殊处理
            except Exception:
                pass
        
        # 检查 LoRA 特有文件
        if (model_path / "adapter_config.json").exists():
            return "bert"  # LoRA 模型
        
        # 检查 BERT 常见文件
        if (model_path / "pytorch_model.bin").exists() or (model_path / "model.safetensors").exists():
            return "bert"
        
        # 默认返回 bert
        return "bert"
    
    def save_model(self, model: NERModel, country: str, description: str = "",
                  metadata: Optional[Dict[str, Any]] = None) -> str:
        """保存模型并登记
        
        参数：
            model: 待保存模型
            country: 国家/区域代码
            description: 模型描述
            metadata: 额外元数据
        
        返回：
            model_id
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{country}_model_{timestamp}"
        model_path = self.model_dir / country / model_name
        model_path.mkdir(parents=True, exist_ok=True)
        
        # 保存模型权重与配置
        model.save_pretrained(str(model_path))
        
        # 可选保存 tokenizer（若模型对象附带）
        # TODO：多数自定义模型未持有 tokenizer；建议在外部保存 tokenizer
        if hasattr(model, 'tokenizer') and model.tokenizer:
            model.tokenizer.save_pretrained(str(model_path))
        
        # 保存自定义配置（包含标签映射与国家信息）
        config_data = {
            'config': getattr(model, 'config_data', {}),
            # ERROR：当 model.config 不是对象而是 dict 时，'.label2id' 会抛出 AttributeError；建议统一对象类型或改为字典访问
            'label2id': getattr(model, 'config', {}).label2id or {},
            'id2label': getattr(model, 'config', {}).id2label or {},
            'country': country,
            'saved_at': datetime.now().isoformat()
        }
        
        config_path = model_path / "config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        # 登记
        model_id = self.register_model(
            str(model_path), country, model.config.model_type if hasattr(model.config, 'model_type') else 'bert',
            description, metadata
        )
        
        self.logger.info(f"Saved model {model_id} to {model_path}")
        return model_id
    
    def list_models(self, country: Optional[str] = None, status: str = "active") -> List[Dict[str, Any]]:
        """列出可用模型
        
        参数：
            country: 过滤国家（可选）
            status: 状态过滤（active/deleted）
        
        返回：
            模型信息列表（按创建时间倒序）
        """
        models = []
        
        for model_id, model_entry in self.registry['models'].items():
            if status and model_entry.get('status') != status:
                continue
            
            if country and model_entry.get('country') != country:
                continue
            
            # 创建模型信息的副本
            model_info = model_entry.copy()
            
            # 添加 LoRA 状态信息
            model_path = Path(model_entry['path'])
            if model_path.exists():
                config_path = model_path / "config.json"
                if config_path.exists():
                    try:
                        with open(config_path, 'r', encoding='utf-8') as f:
                            config_data = json.load(f)
                            model_info['lora_enabled'] = config_data.get('lora_enabled', False)
                            if model_info['lora_enabled']:
                                model_info['lora_config'] = config_data.get('lora_config', {})
                    except Exception:
                        model_info['lora_enabled'] = False
                else:
                    model_info['lora_enabled'] = False
            else:
                model_info['lora_enabled'] = False
            
            models.append(model_info)
        
        # 按创建时间降序
        models.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        return models
    
    def get_model_info(self, model_identifier: str) -> Dict[str, Any]:
        """获取某模型的详细信息
        
        参数：
            model_identifier: 模型 ID 或路径
        
        返回：
            模型信息字典（包含文件系统信息）
        """
        # ID 或路径
        if model_identifier in self.registry['models']:
            model_entry = self.registry['models'][model_identifier].copy()
            model_path = Path(model_entry['path'])
        else:
            model_path = Path(model_identifier)
            model_entry = {
                'id': 'unknown',
                'path': str(model_path),
                'type': self._detect_model_type(str(model_path))
            }
        
        # 文件系统信息
        if model_path.exists():
            model_entry['exists'] = True
            model_entry['size_mb'] = self._get_directory_size(model_path) / (1024 * 1024)
            
            # 读取自定义配置
            config_path = model_path / "config.json"
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                    model_entry['config'] = config_data
                    
                    # 添加 LoRA 信息
                    if config_data.get('lora_enabled', False):
                        model_entry['lora_enabled'] = True
                        model_entry['lora_config'] = config_data.get('lora_config', {})
                    else:
                        model_entry['lora_enabled'] = False
                        
                    # 检查是否存在 LoRA 适配器文件
                    adapter_config_path = model_path / "adapter_config.json"
                    if adapter_config_path.exists():
                        model_entry['has_lora_adapter'] = True
                        try:
                            with open(adapter_config_path, 'r', encoding='utf-8') as f:
                                adapter_config = json.load(f)
                                model_entry['adapter_config'] = adapter_config
                        except Exception as e:
                            self.logger.warning(f"Failed to read adapter config: {e}")
                    else:
                        model_entry['has_lora_adapter'] = False
        else:
            model_entry['exists'] = False
        
        return model_entry
    
    def _get_directory_size(self, path: Path) -> int:
        """计算目录总大小（字节）
        
        # TODO：对大目录频繁调用较耗时，可考虑缓存或异步统计
        """
        total_size = 0
        for file_path in path.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size
    
    def delete_model(self, model_identifier: str, remove_files: bool = True) -> bool:
        """删除模型
        
        参数：
            model_identifier: 模型 ID 或路径
            remove_files: 是否删除文件
        
        返回：
            删除成功返回 True
        """
        try:
            # ID 或路径
            if model_identifier in self.registry['models']:
                model_entry = self.registry['models'][model_identifier]
                model_path = Path(model_entry['path'])
                
                # 标记删除
                self.registry['models'][model_identifier]['status'] = 'deleted'
                self.registry['models'][model_identifier]['deleted_at'] = datetime.now().isoformat()
                self._save_registry()
                
                model_id = model_identifier
            else:
                model_path = Path(model_identifier)
                model_id = model_identifier
            
            # 删除文件
            if remove_files and model_path.exists():
                shutil.rmtree(model_path)
                self.logger.info(f"Removed model files: {model_path}")
            
            self.logger.info(f"Deleted model {model_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete model {model_identifier}: {e}")
            return False
    
    def model_exists(self, model_identifier: str) -> bool:
        """检查模型是否存在（并处于 active 状态）
        
        参数：
            model_identifier: 模型 ID 或路径
        
        返回：
            存在返回 True
        """
        # ID 或路径
        if model_identifier in self.registry['models']:
            model_entry = self.registry['models'][model_identifier]
            if model_entry.get('status') != 'active':
                return False
            model_path = Path(model_entry['path'])
        else:
            model_path = Path(model_identifier)
        
        return model_path.exists()
    
    def get_latest_model(self, country: str) -> Optional[str]:
        """获取某国家/区域的最新模型 ID
        
        参数：
            country: 国家/区域代码
        
        返回：
            最新模型 ID 或 None
        """
        models = self.list_models(country=country)
        
        if models:
            return models[0]['id']  # 已按创建时间倒序
        
        return None
    
    def backup_model(self, model_identifier: str, backup_dir: str) -> str:
        """备份模型到指定目录
        
        参数：
            model_identifier: 模型 ID 或路径
            backup_dir: 备份目录
        
        返回：
            实际备份路径
        """
        # 获取路径
        if model_identifier in self.registry['models']:
            model_entry = self.registry['models'][model_identifier]
            model_path = Path(model_entry['path'])
            model_id = model_identifier
        else:
            model_path = Path(model_identifier)
            model_id = model_path.name
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # 创建备份目录
        backup_path = Path(backup_dir) / f"{model_id}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # 拷贝模型目录
        shutil.copytree(model_path, backup_path / model_path.name)
        
        self.logger.info(f"Backed up model {model_id} to {backup_path}")
        return str(backup_path)
    
    def restore_model(self, backup_path: str, target_path: Optional[str] = None) -> str:
        """从备份恢复模型
        
        参数：
            backup_path: 备份路径
            target_path: 目标恢复路径（可选）
        
        返回：
            实际恢复的模型路径
        """
        backup_path = Path(backup_path)
        
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup not found: {backup_path}")
        
        # 定位备份内的模型目录
        model_dirs = [d for d in backup_path.iterdir() if d.is_dir()]
        if not model_dirs:
            raise ValueError(f"No model directory found in backup: {backup_path}")
        
        model_backup_dir = model_dirs[0]
        
        # 计算目标路径
        if target_path:
            target_path = Path(target_path)
        else:
            target_path = self.model_dir / model_backup_dir.name
        
        # 恢复
        if target_path.exists():
            shutil.rmtree(target_path)
        
        shutil.copytree(model_backup_dir, target_path)
        
        self.logger.info(f"Restored model from {backup_path} to {target_path}")
        return str(target_path)
    
    def cleanup_deleted_models(self) -> int:
        """清理标记为 deleted 的模型（删除其文件并移出注册表）
        
        返回：
            清理的模型数量
        """
        cleaned_count = 0
        
        for model_id, model_entry in list(self.registry['models'].items()):
            if model_entry.get('status') == 'deleted':
                model_path = Path(model_entry['path'])
                
                # 若文件依然存在则删除
                if model_path.exists():
                    shutil.rmtree(model_path)
                    self.logger.info(f"Cleaned up deleted model files: {model_path}")
                
                # 从注册表移除
                del self.registry['models'][model_id]
                cleaned_count += 1
        
        if cleaned_count > 0:
            self._save_registry()
            self.logger.info(f"Cleaned up {cleaned_count} deleted models")
        
        return cleaned_count
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """获取模型统计信息
        
        返回：
            统计字典（数量、类型分布、总体积等）
        """
        stats = {
            'total_models': 0,
            'active_models': 0,
            'deleted_models': 0,
            'countries': set(),
            'model_types': {},
            'lora_models': 0,
            'regular_models': 0,
            'total_size_mb': 0
        }
        
        for model_entry in self.registry['models'].values():
            stats['total_models'] += 1
            
            status = model_entry.get('status', 'active')
            if status == 'active':
                stats['active_models'] += 1
                stats['countries'].add(model_entry.get('country', 'unknown'))
                
                model_type = model_entry.get('type', 'unknown')
                stats['model_types'][model_type] = stats['model_types'].get(model_type, 0) + 1
                
                # 检查是否为 LoRA 模型
                model_path = Path(model_entry['path'])
                if model_path.exists():
                    config_path = model_path / "config.json"
                    if config_path.exists():
                        try:
                            with open(config_path, 'r', encoding='utf-8') as f:
                                config_data = json.load(f)
                                if config_data.get('lora_enabled', False):
                                    stats['lora_models'] += 1
                                else:
                                    stats['regular_models'] += 1
                        except Exception:
                            stats['regular_models'] += 1
                    else:
                        stats['regular_models'] += 1
                    
                    # 计算体积
                    stats['total_size_mb'] += self._get_directory_size(model_path) / (1024 * 1024)
            
            elif status == 'deleted':
                stats['deleted_models'] += 1
        
        stats['countries'] = list(stats['countries'])
        
        return stats
