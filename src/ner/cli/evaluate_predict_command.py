"""基于 predict/evaluate() 的评估命令

特点：
- 使用 `model.predict()` + `NEREvaluator.evaluate()` 执行评估
- 支持 `--confidence-threshold` 模拟线上推理阈值策略
- 与训练解码不同处：低于阈值的词级预测将置为 'O'
"""

import json
from pathlib import Path

from .base import BaseCommand
from ..preprocess import build_preprocessor_from_config

class EvaluatePredictCommand(BaseCommand):
    """基于 predict/evaluate() 的评估命令
    
    特点：
    - 使用 `model.predict()` + `NEREvaluator.evaluate()` 执行评估
    - 支持 `--confidence-threshold` 模拟线上推理阈值策略
    - 与训练解码不同处：低于阈值的词级预测将置为 'O'
    """
    
    @property
    def name(self) -> str:
        return "evaluate-predict"
    
    @property
    def description(self) -> str:
        return "Evaluate using model.predict() + evaluator.evaluate() (serving-like)"
    
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
            "--country", "-c",
            required=True,
            help="Country code for configuration"
        )
        parser.add_argument(
            "--output-dir", "-o",
            type=str,
            help="Output directory for evaluation results"
        )
        parser.add_argument(
            "--confidence-threshold", "-t",
            type=float,
            default=0.5,
            help="Confidence threshold used by model.predict()"
        )
        parser.add_argument(
            "--limit",
            type=int,
            help="Limit number of samples for quick evaluation"
        )
        parser.add_argument(
            "--detailed-report", "--dr",
            action="store_true",
            help="Generate detailed evaluation report (Excel format)"
        )
        parser.add_argument(
            "--entity", "-e",
            type=str,
            help="Comma-separated list of entity types to focus on (e.g., 'COUNTRY,EMIRATE'). If specified, will create a special section for samples where all specified entities have issues."
        )
    
    def execute(self, args) -> bool:
        try:
            import torch
            from transformers import AutoTokenizer
            from ..evaluation import NEREvaluator
            from ..models import NERModelManager
            from ..data import NERDataProcessor
            
            # 加载模型与 tokenizer
            model_manager = NERModelManager(logger=self.logger)
            model = model_manager.load_model(args.model_path)
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
            
            # 解析标签列表（与模型一致）
            if hasattr(model, 'config') and hasattr(model.config, 'id2label') and hasattr(model.config, 'label2id'):
                id2label = model.config.id2label
                try:
                    label_list = [id2label[i] for i in range(len(id2label))]
                except Exception:
                    label_list = list(id2label.values())
            else:
                raise ValueError("Model configuration does not contain label mappings.")
            
            # 加载国家配置与数据
            config = self.get_country_config(args.country)
            processor = NERDataProcessor(config, logger=self.logger)

            # 加载预测数据集
            dataset = processor.load_data_file(args.data_path)

            # 构建预处理器
            preprocessor = build_preprocessor_from_config(config)
            
            # 构造 texts 与 true_labels
            texts = []
            true_labels = [] # 真实标签
            tokens_list = []  # 保存预处理后的tokens列表

            # 遍历数据集，构造 texts 与 true_labels
            for ex in dataset[: (args.limit if getattr(args, 'limit', None) else None)]:
                tokens = ex.get('tokens')
                labels = ex.get('labels')
                if tokens is None or labels is None:
                    continue
                # 使用token-safe预处理，确保标签安全
                proc_tokens, proc_labels = preprocessor.apply_tokens(tokens, labels, allow_non_label_safe=False)
                if not proc_tokens or not proc_labels:
                    continue
                texts.append(' '.join(proc_tokens))
                true_labels.append(proc_labels)
                tokens_list.append(proc_tokens)  # 保存原始tokens列表
            
            if not texts:
                raise ValueError("No valid examples found for evaluation")
            
            # 初始化评估器并执行评估
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            evaluator = NEREvaluator(model, tokenizer, label_list, device, logger=self.logger)
            results = evaluator.evaluate_predict(texts, true_labels, confidence_threshold=getattr(args, 'confidence_threshold', 0.5))
            
            # 打印指标
            evaluator.print_evaluate_results(results)
            
            # 持久化结果
            out_dir = None
            if getattr(args, 'output_dir', None):
                out_dir = Path(args.output_dir)
            elif config and 'output' in config and 'results_dir' in config['output']:
                out_dir = Path(config['output']['results_dir'])
            if out_dir is not None:
                out_dir.mkdir(parents=True, exist_ok=True)
            
            # 生成详细报告
            if getattr(args, 'detailed_report', False):
                self._generate_detailed_report(
                    results, texts, true_labels, tokens_list, config, out_dir, args
                )
            
            return True
        except Exception as e:
            self.logger.error(f"Evaluation (predict) failed: {e}")
            return False
    
    def _generate_detailed_report(self, results, texts, true_labels, tokens_list, config, out_dir, args):
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
            
            predictions = results.get('predictions', [])
            
            if not texts:
                self.logger.warning("No valid examples found for detailed report")
                return
            
            # 确保预测结果与文本数量一致
            if len(predictions) != len(texts):
                self.logger.error(f"Prediction count ({len(predictions)}) != text count ({len(texts)}), truncating")
                return
            
            # 生成报告文件路径
            timestamp = datetime.now().strftime("%Y%m%d%H%M")
            report_filename = f"evaluation_predict_report_{timestamp}.xlsx"
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
                texts, true_labels, predictions, results, entity_types, str(report_path), specified_entities, tokens_list
            )
            
            if report_path:
                self.logger.info(f"Detailed evaluation report generated: {report_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate detailed report: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())