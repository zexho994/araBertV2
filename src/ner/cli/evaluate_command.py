"""评估 NER 模型的命令

说明：
- 当前实现默认从模型目录读取 tokenizer 与标签映射，从而进行文本级评估。
- 支持 `--output-dir` 持久化评估指标 JSON；可选 `--detailed-report` 生成详细报告（未实现）。

# TODO: 支持基于 DataLoader 的批量评估，并尊重 `--batch-size`。
# TODO: 将 `--metrics` 与 `--detailed-report` 真正接入评估与报告逻辑（当前未使用）。
"""

from .base import BaseCommand


class EvaluateCommand(BaseCommand):
    """评估 NER 模型的命令

    说明：
    - 当前实现默认从模型目录读取 tokenizer 与标签映射，从而进行文本级评估。
    - 支持 `--output-dir` 持久化评估指标 JSON；可选 `--detailed-report` 生成详细报告（未实现）。

    # TODO: 支持基于 DataLoader 的批量评估，并尊重 `--batch-size`。
    # TODO: 将 `--metrics` 与 `--detailed-report` 真正接入评估与报告逻辑（当前未使用）。
    """
    
    @property
    def name(self) -> str:
        return "evaluate"
    
    @property
    def description(self) -> str:
        return "Evaluate trained NER models"
    
    def setup_parser(self, parser):
        parser.add_argument(
            "--model-path", "-m",
            type=str,
            required=True,
            help="Path to trained model"
        )
        parser.add_argument(
            "--data-path", "-d",
            type=str,
            required=True,
            help="Path to evaluation data file"
        )
        parser.add_argument(
            "--output-dir", "-o",
            type=str,
            help="Output directory for evaluation results"
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            help="Evaluation batch size"
        )
        parser.add_argument(
            "--country","-c",
            help="Country code for configuration"
        )
        parser.add_argument(
            "--metrics",
            nargs="+",
            default=["precision", "recall", "f1", "accuracy"],
            help="Metrics to compute"
        )
        parser.add_argument(
            "--compare-with",
            help="Compare with another model"
        )
        parser.add_argument(
            "--detailed-report","--dr",
            action="store_true",
            help="Generate detailed evaluation report (Excel format)"
        )
        parser.add_argument(
            "--entity", "-e",
            type=str,
            help="Comma-separated list of entity types to focus on (e.g., 'country'). If specified, will create a special section for samples where all specified entities have issues."
        )
    
    def execute(self, args) -> bool:
        try:
            import torch
            from ..evaluation import NEREvaluator
            from ..models import NERModelManager
            from pathlib import Path
            from ..data import NERDataProcessor, NERDataLoader
            from transformers import AutoTokenizer

            # 加载国家配置（用于数据与标签与LoRA设置一致）
            if not getattr(args, 'country', None):
                raise ValueError("--country is required for evaluate to ensure consistent data processing")
            config = self.get_country_config(args.country)

            # 根据配置决定加载普通模型或LoRA组合模型
            lora_cfg = (config.get('training', {}) or {}).get('lora', {}) or {}
            use_lora = bool(lora_cfg.get('enabled', False))

            if use_lora:
                # LoRA评估：使用 base + adapter 组合
                from ..models import BertNERModel
                try:
                    from peft import PeftModel  # type: ignore
                except Exception as e:
                    raise RuntimeError(f"LoRA evaluation requires 'peft' package: {e}")

                base_model_path = (config.get('model', {}) or {}).get('pretrained_model') or args.model_path
                adapter_path = (config.get('model', {}) or {}).get('lora_adapter')
                if not adapter_path:
                    raise ValueError("LoRA enabled but 'model.lora_adapter' not set in config")

                # 标签映射来自配置，确保与训练一致
                labels_cfg = (config.get('labels', {}) or {})
                model_label2id = labels_cfg.get('label_mapping', {})
                if not model_label2id:
                    raise ValueError("Config 'labels.label_mapping' is required when using LoRA evaluate")
                id2label_map = {int(v): k for k, v in model_label2id.items()}
                label_list = [id2label_map[i] for i in range(len(id2label_map))]

                # 构建基础模型并加载LoRA适配器
                model = BertNERModel.from_pretrained(
                    pretrained_model_name_or_path=base_model_path,
                    num_labels=len(model_label2id),
                    dropout=(config.get('model', {}) or {}).get('dropout', 0.1),
                    label2id=model_label2id,
                    id2label=id2label_map
                )
                model = PeftModel.from_pretrained(model, adapter_path)
                tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)
                tokenizer_name_for_loader = base_model_path
            else:
                # 常规评估：从训练产物目录加载完整模型
                model_manager = NERModelManager(logger=self.logger)
                model = model_manager.load_model(args.model_path)

                # 加载 tokenizer（保持与训练一致，优先从模型目录加载）
                tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)

                # 从模型配置中获取标签列表与映射（确保与模型训练时一致）
                if hasattr(model, 'config') and hasattr(model.config, 'id2label') and hasattr(model.config, 'label2id'):
                    id2label = model.config.id2label
                    model_label2id = model.config.label2id
                    try:
                        label_list = [id2label[i] for i in range(len(id2label))]
                    except Exception as e:
                        self.logger.error(f"加载标签列表与映射发生回退, ID2Label: {id2label}, Label2ID: {model_label2id}: {e}")
                        label_list = list(id2label.values())
                else:
                    raise ValueError("Model configuration does not contain label mappings.")
                tokenizer_name_for_loader = args.model_path

            # 准备数据（与训练流程一致）
            processor = NERDataProcessor(config, logger=self.logger)
            val_dataset_file = processor.load_data_file(args.data_path)

            # 构建与训练一致的 DataLoader（使用 is_split_into_words 对齐）
            ner_loader_builder = NERDataLoader(
                tokenizer_name=tokenizer_name_for_loader,
                label2id=model_label2id,
                max_length=config.get('data', {}).get('max_length', 512),
                logger=self.logger
            )
            val_dataset = ner_loader_builder.create_dataset_loader(val_dataset_file)
            batch_size = getattr(args, 'batch_size', None) or config.get('training', {}).get('batch_size', 16)
            val_loader = ner_loader_builder.create_dataloader(val_dataset, batch_size=batch_size, shuffle=False)

            # 初始化评估器，迁移模型至设备
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            evaluator = NEREvaluator(model, tokenizer, label_list, device, logger=self.logger)

            # 运行基于 DataLoader 的评估
            results = evaluator.evaluate_dataloader(val_loader)
            
            # 打印评估结果
            evaluator.print_evaluate_results(results)

            # 持久化结果
            out_dir = None
            if getattr(args, 'output_dir', None):
                out_dir = Path(args.output_dir)
            elif config and 'output' in config and 'results_dir' in config['output']:
                out_dir = Path(config['output']['results_dir'])
            
            # 生成详细报告
            if getattr(args, 'detailed_report', False):
                self._generate_detailed_report(
                    results, val_dataset_file, config, out_dir, args
                )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            return False
    
    def _generate_detailed_report(self, results, val_dataset, config, out_dir, args):
        """生成详细的评估报告"""
        try:
            from ..evaluation.report_generator import NERReportGenerator
            from datetime import datetime

            self.logger.info(f"Starting to generate detailed report to: {out_dir}")
            
            # 获取实体类型列表
            label_mapping = config.get('labels', {}).get('label_mapping', {})
            if not label_mapping:
                self.logger.warning("No label mapping found in config, skipping detailed report")
                return
            
            # 从BIO标签中提取实体类型（去掉B-/I-前缀）
            entity_types = set()
            for label in label_mapping.keys():
                if label != 'O' and '-' in label:
                    entity_type = label.split('-', 1)[1]  # 去掉B-/I-前缀
                    entity_types.add(entity_type)
            entity_types = sorted(list(entity_types))
            
            # 如果没有实体类型，则跳过详细报告
            if not entity_types:
                self.logger.warning("No entity types found in label mapping, skipping detailed report")
                return
            
            self.logger.info(f"Extracted entity types: {entity_types}")
            
            texts = []
            true_labels = []
            predictions = results.get('predictions', [])
            
            for ex in val_dataset:
                tokens = ex.get('tokens')
                labels = ex.get('labels')
                if tokens is None or labels is None:
                    continue
                
                texts.append(' '.join(tokens))
                true_labels.append(labels)
            
            if not texts:
                self.logger.warning("No valid examples found for detailed report")
                return
            
            # 确保预测结果与文本数量一致
            if len(predictions) != len(texts):
                self.logger.error(f"Prediction count ({len(predictions)}) != text count ({len(texts)}), truncating")
                return
            
            # 生成报告文件路径
            timestamp = datetime.now().strftime("%Y%m%d%H%M")
            report_filename = f"evaluation_report_{timestamp}.xlsx"
            report_path = out_dir / report_filename if out_dir else report_filename
            
            # 解析指定的实体类型
            specified_entities = None
            if hasattr(args, 'entity') and args.entity:
                specified_entities = [e.strip() for e in args.entity.split(',')]
                # 验证指定的实体类型是否存在于配置中
                invalid_entities = [e for e in specified_entities if e not in entity_types]
                if invalid_entities:
                    self.logger.warning(f"Invalid entity types specified: {invalid_entities}. Available types: {entity_types}")
                    specified_entities = [e for e in specified_entities if e in entity_types]
                if specified_entities:
                    self.logger.info(f"Focusing on entity types: {specified_entities}")
            
            # 生成Excel报告
            report_generator = NERReportGenerator(logger=self.logger)
            report_path = report_generator.generate_excel_report(
                texts, true_labels, predictions, results, entity_types, str(report_path), specified_entities
            )
            
            if report_path:
                self.logger.info(f"Detailed evaluation report generated: {report_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate detailed report: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
