"""NER 配置管理器（Configuration Manager）

职责：
- 加载/校验/管理按国家划分的配置文件
- 统一访问配置模板与外部模板
- 提供创建国家配置与工程目录/样例数据的能力

设计说明：
- 默认配置根目录为 `data/ner/configs/`，下含 `countries/` 与 `templates/`
- 国家配置当前仅支持 `.json`，模板支持 `.json5`（若已安装 json5）与 `.json`

# TODO: 为国家配置增加 `.json5` 支持，复用模板解析逻辑（优先 json5，其次 json）。
# TODO: 提供更严格的 Schema 校验（可对接 pydantic/voluptuous）并输出聚合错误。
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from copy import deepcopy

# Try to import json5 for enhanced JSON parsing with comments support
try:
    import json5
    HAS_JSON5 = True
except ImportError:
    json5 = None
    HAS_JSON5 = False

# 默认配置根目录
DEFAULT_CONFIG_DIR = "data/ner/configs"
DEFAULT_COUNTRY_CONFIG_DIR = DEFAULT_CONFIG_DIR + "/countries"
DEFAULT_TEMPLATE_CONFIG_DIR = DEFAULT_CONFIG_DIR + "/templates"

class ConfigManager:
    """NER 配置文件与设置的统一管理入口

    - 负责配置的读取/缓存/写入
    - 提供模板加载与国家配置创建
    - 负责工程目录与样例数据的初始化

    # TODO: 支持按环境（dev/test/prod）或版本管理配置（如带有 `version` 字段）。
    """
    
    def __init__(self, config_dir: str = DEFAULT_CONFIG_DIR):
        """初始化配置管理器
        
        Args:
            config_dir: 存放配置文件的根目录

        注意：
        - 构造时会确保 `countries/` 与 `templates/` 目录存在。
        - 为提升健壮性，建议在上层传入绝对路径或项目根相对路径。

        # TODO: 若在只读环境下应避免自动创建目录，或提供显式开关。
        """
        self.config_dir = Path(config_dir)
        self.countries_dir = Path(DEFAULT_COUNTRY_CONFIG_DIR)
        self.templates_dir = Path(DEFAULT_TEMPLATE_CONFIG_DIR)
        self._config_cache = {}
        
        # Ensure directories exist
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.countries_dir.mkdir(parents=True, exist_ok=True)
        self.templates_dir.mkdir(parents=True, exist_ok=True)
    
    def load_country_config(self, country: str, use_cache: bool = True) -> Dict[str, Any]:
        """加载指定国家的配置
        
        Args:
            country: 国家代码（如 'uae', 'saudi'）
            use_cache: 是否优先使用缓存
            
        Returns:
            该国家的配置字典
            
        Raises:
            FileNotFoundError: 对应配置文件不存在
            ValueError: 配置解析或校验失败

        # TODO: 支持 `.json5` 的国家配置文件加载；如同时存在，以 `.json5` 优先。
        # TODO: 提供基于 mtime 的缓存失效策略，或允许外部显式禁用缓存。
        """
        if use_cache and country in self._config_cache:
            return deepcopy(self._config_cache[country])
        
        config_file = self.countries_dir / f"{country}.json"
        
        if not config_file.exists():
            raise FileNotFoundError(
                f"Configuration file for country '{country}' not found at {config_file}"
            )
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Validate configuration
            self._validate_config(config, country)
            
            # Cache the configuration
            self._config_cache[country] = deepcopy(config)
            
            return config
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file {config_file}: {e}")
        except Exception as e:
            raise ValueError(f"Error loading configuration for {country}: {e}")
    
    def load_template_config(self, template: str) -> Dict[str, Any]:
        """加载模板配置
        
        Args:
            template: 模板名（如 'default', 'address_ner'）
            
        Returns:
            模板配置字典

        说明：
        - 若安装了 json5，优先尝试读取 `.json5`；否则回退 `.json`。
        """
        # Try to load .json5 file first if json5 is available
        if HAS_JSON5:
            json5_file = self.templates_dir / f"{template}.json5"
            if json5_file.exists():
                try:
                    with open(json5_file, 'r', encoding='utf-8') as f:
                        return json5.load(f)
                except Exception as e:
                    raise ValueError(f"Invalid JSON5 in template file {json5_file}: {e}")
        
        # Fall back to .json file
        template_file = self.templates_dir / f"{template}.json"
        
        if not template_file.exists():
            raise FileNotFoundError(
                f"Template '{template}' not found at {template_file} or {template}.json5"
            )
        
        try:
            with open(template_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in template file {template_file}: {e}")
    
    def load_external_template(self, template_path: str) -> Dict[str, Any]:
        """从外部路径加载模板配置
        
        Args:
            template_path: 模板文件绝对路径
            
        Returns:
            模板配置字典

        # TODO: 对 `.jsonl` 的处理当前仅读取首行；可考虑支持全量/多行合并。
        """
        template_file = Path(template_path)
        
        if not template_file.exists():
            raise FileNotFoundError(
                f"External template not found at {template_file}"
            )
        
        # Determine file format based on extension
        if template_file.suffix.lower() == '.json5' and HAS_JSON5:
            try:
                with open(template_file, 'r', encoding='utf-8') as f:
                    return json5.load(f)
            except Exception as e:
                raise ValueError(f"Invalid JSON5 in external template file {template_file}: {e}")
        elif template_file.suffix.lower() in ['.json', '.jsonl']:
            try:
                with open(template_file, 'r', encoding='utf-8') as f:
                    # Handle JSONL format (load first line only)
                    if template_file.suffix.lower() == '.jsonl':
                        first_line = f.readline().strip()
                        if not first_line:
                            raise ValueError("JSONL file is empty")
                        return json.loads(first_line)
                    else:
                        return json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in external template file {template_file}: {e}")
        else:
            raise ValueError(
                f"Unsupported template file format: {template_file.suffix}. "
                f"Supported formats: .json, .json5{', .jsonl' if HAS_JSON5 else ''}"
            )
    
    def save_country_config(self, country: str, config: Dict[str, Any]) -> None:
        """保存国家配置
        
        Args:
            country: 国家代码
            config: 待持久化的配置字典

        # TODO: 使用原子写（临时文件 + 覆盖）避免中途失败导致的文件损坏。
        # TODO: 可选开启 JSON 排序键与结尾换行，增强可读性与稳定性。
        """
        # Validate configuration before saving
        self._validate_config(config, country)
        
        config_file = self.countries_dir / f"{country}.json"
        
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            # Update cache
            self._config_cache[country] = deepcopy(config)
            
        except Exception as e:
            raise ValueError(f"Error saving configuration for {country}: {e}")
    
    def create_country_config(self, country: str, template: str = "default", 
                                external_template_path: Optional[str] = None) -> Dict[str, Any]:
        """基于模板创建新的国家配置
        
        Args:
            country: 国家代码
            template: 模板名称（若指定 external_template_path 则忽略）
            external_template_path: 外部模板文件路径（可选）
            
        Returns:
            新创建的配置字典

        说明：
        - 若使用 DAPT 模板，会调用 `_fix_template_structure` 调整为 NER 兼容结构。
        - 会自动进行占位符替换与项目结构初始化。
        """
        # Load template from external path or standard template
        if external_template_path:
            config = self.load_external_template(external_template_path)
            # Fix DAPT template structure for NER compatibility
            self._fix_template_structure(config)
        else:
            config = self.load_template_config(template)
        
        # Apply placeholder replacement
        config = self._replace_placeholders(config, country)
        
        # Update country-specific information (in case placeholders weren't used)
        if "country" in config:
            config["country"]["code"] = country
            config["country"]["name"] = country.upper()
        
        # Save the new configuration
        self.save_country_config(country, config)
        
        # Create directories and files based on configuration
        self._create_project_structure(config, country)
        
        return config
    
    def list_countries(self) -> List[str]:
        """列出当前所有可用的国家配置
        
        Returns:
            国家代码列表（按名称排序）
        """
        if not self.countries_dir.exists():
            return []
        
        countries = []
        for config_file in self.countries_dir.glob("*.json"):
            countries.append(config_file.stem)
        
        return sorted(countries)
    
    def list_templates(self) -> List[str]:
        """列出所有可用的配置模板
        
        Returns:
            模板名称列表（去重并排序）
        """
        if not self.templates_dir.exists():
            return []
        
        templates = set()
        
        # Add .json5 templates if json5 is available
        if HAS_JSON5:
            for template_file in self.templates_dir.glob("*.json5"):
                templates.add(template_file.stem)
        
        # Add .json templates
        for template_file in self.templates_dir.glob("*.json"):
            templates.add(template_file.stem)
        
        return sorted(list(templates))
    
    def country_exists(self, country: str) -> bool:
        """检查指定国家配置是否存在
        
        Args:
            country: 要检查的国家代码
            
        Returns:
            存在返回 True，否则 False
        """
        config_file = self.countries_dir / f"{country}.json"
        return config_file.exists()
    
    def delete_country_config(self, country: str) -> None:
        """删除指定国家的配置
        
        Args:
            country: 待删除的国家代码

        # TODO: 支持回收站/备份机制，避免误删；或增加 `force`/交互确认机制在 CLI 层实现。
        """
        config_file = self.countries_dir / f"{country}.json"
        
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration for country '{country}' not found")
        
        config_file.unlink()
        
        # Remove from cache
        if country in self._config_cache:
            del self._config_cache[country]
    
    def clear_cache(self) -> None:
        """清空配置缓存"""
        self._config_cache.clear()
    
    def _replace_placeholders(self, config: Dict[str, Any], country: str) -> Dict[str, Any]:
        """替换模板中的占位符
        
        Args:
            config: 含占位符的配置字典
            country: 用于替换的国家代码
            
        Returns:
            替换占位符后的配置字典

        # TODO: 支持更多占位符（如日期/作者/版本等），并允许自定义替换上下文。
        """
        # Convert config to JSON string for placeholder replacement
        config_str = json.dumps(config, ensure_ascii=False, indent=2)
        
        # Generate country name from country code (capitalize and replace underscores)
        country_name = country.replace('_', ' ').title()
        
        # Replace placeholders
        config_str = config_str.replace('{country_code}', country)
        config_str = config_str.replace('{country_name}', country_name)
        
        # Parse back to dictionary
        try:
            return json.loads(config_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing configuration after placeholder replacement: {e}")
    
    def _fix_template_structure(self, config: Dict[str, Any]) -> None:
        """修正模板结构以兼容 NER 配置
        
        作用：
        - 处理 DAPT 与 NER 模板结构上的差异
        - 将 `model.num_labels` 移动到 `labels.num_labels`
        - 若缺失 `model.type`，默认补为 'bert'
        
        Args:
            config: 待修正的配置字典（原地修改）

        # TODO: 补充更多字段的兼容映射（如 `training` 字段命名差异）。
        """
        # Fix model section fields
        if 'model' in config:
            model_config = config['model']
            
            # Convert base_model to name and pretrained_model
            if 'base_model' in model_config and 'name' not in model_config:
                model_config['name'] = model_config['base_model'].split('/')[-1]  # Extract model name
                model_config['pretrained_model'] = model_config['base_model']
            
            # Add missing type field
            if 'type' not in model_config:
                model_config['type'] = 'bert'  # Default to bert type
        
        # Check if num_labels is in model section but missing from labels section
        if ('model' in config and 'num_labels' in config['model'] and 
            'labels' in config and 'num_labels' not in config['labels']):
            
            # Move num_labels from model to labels section
            config['labels']['num_labels'] = config['model']['num_labels']
            
        # If labels section exists but num_labels is still missing, calculate it
        if ('labels' in config and 'num_labels' not in config['labels'] and 
            'label_names' in config['labels']):
            
            config['labels']['num_labels'] = len(config['labels']['label_names'])
    
    def _create_project_structure(self, config: Dict[str, Any], country: str) -> None:
        """根据配置创建工程目录与样例数据
        
        Args:
            config: 配置字典
            country: 国家代码

        说明：
        - 会根据 `output.*`、`logging.log_file`、`data.{train,val,test}_file` 创建目录/样例文件

        # TODO: 支持自定义样例数据模板；对于已存在文件提供覆盖/跳过开关。
        """
        import logging
        logger = logging.getLogger(__name__)
        
        created_dirs = []
        created_files = []
        
        try:
            # Create directories from configuration paths
            dirs_to_create = []
            
            # Extract directory paths from configuration
            if 'output' in config:
                output_config = config['output']
                if 'model_dir' in output_config:
                    dirs_to_create.append(Path(output_config['model_dir']))
                if 'results_dir' in output_config:
                    dirs_to_create.append(Path(output_config['results_dir']))
                if 'logs_dir' in output_config:
                    dirs_to_create.append(Path(output_config['logs_dir']))
            
            if 'logging' in config and 'log_file' in config['logging']:
                log_file_path = Path(config['logging']['log_file'])
                dirs_to_create.append(log_file_path.parent)
            
            # Extract data file paths and their directories
            data_files = []
            if 'data' in config:
                data_config = config['data']
                for file_key in ['train_file', 'val_file', 'test_file']:
                    if file_key in data_config:
                        file_path = Path(data_config[file_key])
                        dirs_to_create.append(file_path.parent)
                        data_files.append((file_key, file_path))
            
            # Create directories
            for dir_path in dirs_to_create:
                if not dir_path.exists():
                    dir_path.mkdir(parents=True, exist_ok=True)
                    created_dirs.append(str(dir_path))
                    logger.info(f"Created directory: {dir_path}")
            
            # Create sample data files
            for file_key, file_path in data_files:
                if not file_path.exists():
                    self._create_sample_data_file(file_path, file_key, config)
                    created_files.append(str(file_path))
                    logger.info(f"Created sample data file: {file_path}")
            
            # Log summary
            if created_dirs or created_files:
                logger.info(f"Project structure created for country '{country}':")
                if created_dirs:
                    logger.info(f"  Directories: {len(created_dirs)} created")
                if created_files:
                    logger.info(f"  Data files: {len(created_files)} created")
            else:
                logger.info(f"Project structure already exists for country '{country}'")
                
        except Exception as e:
            logger.error(f"Error creating project structure for {country}: {e}")
            raise
    
    def _create_sample_data_file(self, file_path: Path, file_type: str, config: Dict[str, Any]) -> None:
        """创建带有 NER 标注的样例数据（JSONL）
        
        Args:
            file_path: 文件输出路径
            file_type: 文件类型（train_file/val_file/test_file）
            config: 配置字典（可用于读取标签）

        # TODO: 允许按 `label_names` 生成更贴合业务的样例；支持多语言示例。
        """
        # Sample NER data in JSONL format
        sample_data = []
        
        # Get label names from config, or use default labels
        label_names = []
        if 'labels' in config and 'label_names' in config['labels']:
            label_names = config['labels']['label_names']
        
        # Default NER labels if not specified
        if not label_names:
            label_names = [
                "O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC",
                "B-MISC", "I-MISC", "B-ADDR", "I-ADDR", "B-STREET", "I-STREET",
                "B-CITY", "I-CITY", "B-STATE", "I-STATE"
            ]
        
        # Sample sentences with NER annotations
        if file_type == 'train_file':
            sample_data = [
                {
                    "id": "train_001",
                    "tokens": ["John", "Smith", "lives", "in", "New", "York", "City", "."],
                    "labels": ["B-PER", "I-PER", "O", "O", "B-LOC", "I-LOC", "I-LOC", "O"]
                },
                {
                    "id": "train_002",
                    "tokens": ["Apple", "Inc", "is", "located", "at", "123", "Main", "Street", "."],
                    "labels": ["B-ORG", "I-ORG", "O", "O", "O", "B-ADDR", "I-ADDR", "I-ADDR", "O"]
                },
                {
                    "id": "train_003",
                    "tokens": ["Visit", "the", "Emirates", "Palace", "in", "Abu", "Dhabi", "."],
                    "labels": ["O", "O", "B-LOC", "I-LOC", "O", "B-CITY", "I-CITY", "O"]
                }
            ]
        elif file_type == 'val_file':
            sample_data = [
                {
                    "id": "val_001",
                    "tokens": ["Microsoft", "Corporation", "headquarters", "in", "Seattle", "."],
                    "labels": ["B-ORG", "I-ORG", "O", "O", "B-CITY", "O"]
                },
                {
                    "id": "val_002",
                    "tokens": ["Dr", ".", "Ahmed", "works", "at", "Dubai", "Hospital", "."],
                    "labels": ["B-PER", "I-PER", "I-PER", "O", "O", "B-ORG", "I-ORG", "O"]
                }
            ]
        else:  # test_file
            sample_data = [
                {
                    "id": "test_001",
                    "tokens": ["The", "meeting", "is", "at", "456", "Oak", "Avenue", "."],
                    "labels": ["O", "O", "O", "O", "B-ADDR", "I-ADDR", "I-ADDR", "O"]
                },
                {
                    "id": "test_002",
                    "tokens": ["Google", "LLC", "in", "Mountain", "View", "California", "."],
                    "labels": ["B-ORG", "I-ORG", "O", "B-CITY", "I-CITY", "B-STATE", "O"]
                }
            ]
        
        # Write JSONL file
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                for item in sample_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        except Exception as e:
            raise ValueError(f"Error creating sample data file {file_path}: {e}")
    
    def _validate_config(self, config: Dict[str, Any], country: str) -> None:
        """校验配置结构与必需字段
        
        Args:
            config: 待校验的配置
            country: 国家代码（用于错误提示）
            
        Raises:
            ValueError: 配置不合法

        说明：
        - 当前强制要求 `labels` 包含 `num_labels`/`label_names`/`label_mapping`；
          与训练器中允许通过 `entities` 推导 BIO 标签的逻辑存在一定冗余。

        # TODO: 统一标签定义来源与约束，避免“配置校验通过但训练时重建标签”的重复逻辑。
        # TODO: 校验 `label_mapping` 与 `label_names` 的对应关系与口径（BIO vs 非 BIO）。
        """
        required_sections = [
            'country', 'model', 'training', 'data', 'labels', 
            'evaluation', 'output', 'hardware', 'logging'
        ]
        
        for section in required_sections:
            if section not in config:
                raise ValueError(
                    f"Missing required section '{section}' in {country} configuration"
                )
        
        # Validate country section
        country_config = config['country']
        if 'code' not in country_config or 'name' not in country_config:
            raise ValueError("Country section must contain 'code' and 'name' fields")
        
        # Validate labels section
        labels_config = config['labels']
        required_label_fields = ['num_labels', 'label_names', 'label_mapping']
        for field in required_label_fields:
            if field not in labels_config:
                raise ValueError(f"Labels section missing required field '{field}'")
        
        # Validate label consistency
        num_labels = labels_config['num_labels']
        label_names = labels_config['label_names']
        label_mapping = labels_config['label_mapping']
        
        if len(label_names) != num_labels:
            raise ValueError(
                f"Number of label names ({len(label_names)}) doesn't match num_labels ({num_labels})"
            )
        
        if len(label_mapping) != num_labels:
            raise ValueError(
                f"Number of label mappings ({len(label_mapping)}) doesn't match num_labels ({num_labels})"
            )
        
        # Validate model section
        model_config = config['model']
        required_model_fields = ['name', 'type', 'pretrained_model']
        for field in required_model_fields:
            if field not in model_config:
                raise ValueError(f"Model section missing required field '{field}'")
    
    def get_config_summary(self, country: str) -> Dict[str, Any]:
        """获取指定国家配置的摘要信息
        
        Args:
            country: 国家代码
            
        Returns:
            配置摘要字典（核心字段集中展示）

        # TODO: 支持更多摘要字段（如数据路径、warmup、scheduler 等）。
        """
        config = self.load_country_config(country)
        
        return {
            'country': config['country'],
            'model_name': config['model']['name'],
            'model_type': config['model']['type'],
            'num_labels': config['labels']['num_labels'],
            'training_epochs': config['training']['epochs'],
            'batch_size': config['training']['batch_size'],
            'learning_rate': config['training']['learning_rate']
        }