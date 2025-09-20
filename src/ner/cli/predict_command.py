"""使用已训练模型进行预测的命令

支持：
- 单条文本预测（--text）
- 文件批量预测（--file，每行一条文本）
- 多种输出格式（json/text/conll）

# TODO: 支持输入 JSON/JSONL（含 tokens/labels）并保持结构化输出。
"""

import json

from .base import BaseCommand
from ..preprocess import build_preprocessor_from_config

class PredictCommand(BaseCommand):
    """使用已训练模型进行预测的命令

    支持：
    - 单条文本预测（--text）
    - 文件批量预测（--file，每行一条文本）
    - 多种输出格式（json/text/conll）

    # TODO: 支持输入 JSON/JSONL（含 tokens/labels）并保持结构化输出。
    """
    
    @property
    def name(self) -> str:
        return "predict"
    
    @property
    def description(self) -> str:
        return "Use trained model for prediction"
    
    def setup_parser(self, parser):
        parser.add_argument(
            "--model-path",
            required=False,
            help="Model name or path for prediction"
        )
        parser.add_argument(
            "--text",
            required=False,
            help="Text to predict (single prediction)"
        )
        parser.add_argument(
            "--file",
            required=False,
            help="File containing texts to predict"
        )
        parser.add_argument(
            "--output-file",
            required=False,
            help="File to save predictions"
        )
        parser.add_argument(
            "--output-format",
            choices=["json", "text", "conll"],
            default="json",
            help="Output format"
        )
        parser.add_argument(
            "--confidence-threshold",
            type=float,
            default=0.5,
            help="Confidence threshold for predictions"
        )
    
    def execute(self, args) -> bool:
        try:
            # 若既未提供 --text 也未提供 --file，则进入交互式 REPL
            if not getattr(args, 'text', None) and not getattr(args, 'file', None):
                from .repl_predict import PredictREPL
                repl = PredictREPL(logger=self.logger)
                # 继承一次性参数作为默认会话配置
                if getattr(args, 'output_format', None):
                    repl.output_format = args.output_format
                if getattr(args, 'confidence_threshold', None) is not None:
                    repl.confidence_threshold = args.confidence_threshold
                repl.run()
                return True

            # 执行一次性预测路径：需要提供 model_path
            if not getattr(args, 'model_path', None):
                self.logger.error("--model-path is required when using --text or --file")
                return False

            # 抑制 transformers 警告（仅在执行一次性预测时才导入重库）
            import logging
            logging.getLogger("transformers").setLevel(logging.ERROR)

            # 懒加载重依赖
            from ..models import NERModelManager
            from transformers import AutoTokenizer

            # 加载模型与分词器
            model_manager = NERModelManager(self.global_config.get_model_dir(), logger=self.logger)
            model = model_manager.load_model(args.model_path)
            tokenizer = AutoTokenizer.from_pretrained(args.model_path)

            # 构建预处理器
            preprocessor = None
            try:
                if self.get_country_config(args.country):
                    preprocessor = build_preprocessor_from_config(self.get_country_config(args.country))
                else:
                    # 兜底：使用空配置
                    preprocessor = build_preprocessor_from_config({})
            except Exception:
                preprocessor = None

            # 单条文本预测
            if args.text:
                input_text = args.text
                if preprocessor:
                    input_text = preprocessor.apply_text(input_text)
                prediction = model.predict(input_text, tokenizer=tokenizer, confidence_threshold=args.confidence_threshold)
                
                if args.output_format == "json":
                    self.logger.info(json.dumps(prediction, indent=2, ensure_ascii=False))
                elif args.output_format == "text":
                    for token, label in zip(prediction['tokens'], prediction['labels']):
                        self.logger.info(f"{token}\t{label}")
                elif args.output_format == "conll":
                    for token, label in zip(prediction['tokens'], prediction['labels']):
                        self.logger.info(f"{token} {label}")
                return True
            
            if args.file:
                predictions = []
                with open(args.file, 'r', encoding='utf-8') as f:
                    for line in f:
                        text = line.strip()
                        if text:
                            if preprocessor:
                                text = preprocessor.apply_text(text)
                            prediction = model.predict(text, tokenizer=tokenizer, confidence_threshold=args.confidence_threshold)
                            predictions.append(prediction)
                
                if args.output_file:
                    with open(args.output_file, 'w', encoding='utf-8') as f:
                        if args.output_format == "json":
                            json.dump(predictions, f, indent=2, ensure_ascii=False)
                        else:
                            for pred in predictions:
                                if args.output_format == "text":
                                    for token, label in zip(pred['tokens'], pred['labels']):
                                        f.write(f"{token}\t{label}\n")
                                    f.write("\n")
                                elif args.output_format == "conll":
                                    for token, label in zip(pred['tokens'], pred['labels']):
                                        f.write(f"{token} {label}\n")
                                    f.write("\n")
                else:
                    for pred in predictions:
                        self.logger.info(json.dumps(pred, indent=2, ensure_ascii=False))
                
                return True
            
            self.logger.error("Please provide either --text or --file")
            return False
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return False
