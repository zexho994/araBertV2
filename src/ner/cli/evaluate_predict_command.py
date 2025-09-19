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
            "--country",
            required=True,
            help="Country code for configuration"
        )
        parser.add_argument(
            "--output-dir", "-o",
            type=str,
            help="Output directory for evaluation results"
        )
        parser.add_argument(
            "--confidence-threshold",
            type=float,
            default=0.5,
            help="Confidence threshold used by model.predict()"
        )
        parser.add_argument(
            "--limit",
            type=int,
            help="Limit number of samples for quick evaluation"
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
            dataset = processor.load_data_file(args.data_path)

            # 构建预处理器
            preprocessor = build_preprocessor_from_config(config)
            
            # 构造 texts 与 true_labels
            texts = []
            true_labels = []
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
            
            if not texts:
                raise ValueError("No valid examples found for evaluation")
            
            # 初始化评估器并执行评估
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            evaluator = NEREvaluator(model, tokenizer, label_list, device, logger=self.logger)
            results = evaluator.evaluate(texts, true_labels, confidence_threshold=getattr(args, 'confidence_threshold', 0.5))
            
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
                metrics_path = out_dir / 'metrics_predict.json'
                with open(metrics_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                self.logger.info(f"\nSaved evaluation (predict) metrics to: {metrics_path}")
            
            return True
        except Exception as e:
            self.logger.error(f"Evaluation (predict) failed: {e}")
            return False
