"""使用已训练模型进行预测的命令

支持：
- 单条文本预测（--text）
- 文件批量预测（--file，每行一条文本，或JSONL格式含tokens/labels）
- 多种输出格式（json/text/conll）
- 生成详细的预测结果报告（--detailed-report，Excel格式）
- 支持JSONL格式输入，自动对比预测与真实标签
"""

import json

from .base import BaseCommand
from ..preprocess import build_preprocessor_from_config

class PredictCommand(BaseCommand):
    """使用已训练模型进行预测的命令

    支持：
    - 单条文本预测（--text）
    - 文件批量预测（--file，每行一条文本，或JSONL格式含tokens/labels）
    - 多种输出格式（json/text/conll）
    - 生成详细的预测结果报告（--detailed-report，Excel格式）
    - 支持JSONL格式输入，自动对比预测与真实标签
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
        parser.add_argument(
            "--detailed-report", "--dr",
            action="store_true",
            help="Generate detailed prediction report (Excel format) for batch prediction"
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
                original_texts = []
                true_labels_list = []  # 存储真实标签（如果输入是JSONL格式）
                tokens_list = []  # 存储tokens（如果输入是JSONL格式）
                is_jsonl = False
                
                # 读取文件并判断格式
                with open(args.file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        
                        # 尝试解析为JSON
                        try:
                            data = json.loads(line)
                            if isinstance(data, dict) and 'text' in data:
                                # JSONL格式
                                is_jsonl = True
                                text = data['text']
                                tokens = data.get('tokens', [])
                                true_labels = data.get('labels', [])
                                
                                original_texts.append(text)
                                tokens_list.append(tokens)
                                true_labels_list.append(true_labels)
                                
                                # 使用text进行预测
                                processed_text = text
                                if preprocessor:
                                    processed_text = preprocessor.apply_text(text)
                                prediction = model.predict(processed_text, tokenizer=tokenizer, confidence_threshold=args.confidence_threshold)
                                predictions.append(prediction)
                            else:
                                self.logger.warning(f"Line {line_num}: Invalid JSONL format, treating as plain text")
                                is_jsonl = False
                                original_texts.append(line)
                                tokens_list.append([])
                                true_labels_list.append([])
                                processed_text = line
                                if preprocessor:
                                    processed_text = preprocessor.apply_text(line)
                                prediction = model.predict(processed_text, tokenizer=tokenizer, confidence_threshold=args.confidence_threshold)
                                predictions.append(prediction)
                        except json.JSONDecodeError:
                            # 普通文本格式
                            original_texts.append(line)
                            tokens_list.append([])
                            true_labels_list.append([])
                            processed_text = line
                            if preprocessor:
                                processed_text = preprocessor.apply_text(line)
                            prediction = model.predict(processed_text, tokenizer=tokenizer, confidence_threshold=args.confidence_threshold)
                            predictions.append(prediction)
                
                # 生成详细报告
                if getattr(args, 'detailed_report', False):
                    # 检查是否有真实标签（JSONL格式且包含labels）
                    has_true_labels = is_jsonl and any(true_labels_list)
                    
                    if has_true_labels:
                        # 生成评估式报告（对比预测与真实标签）
                        self._generate_comparison_report(
                            original_texts, tokens_list, true_labels_list, predictions, args
                        )
                    else:
                        # 生成普通预测报告
                        self._generate_prediction_report(
                            original_texts, predictions, args
                        )
                
                return True
            
            self.logger.error("Please provide either --text or --file")
            return False
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return False
    
    def _generate_comparison_report(self, texts, tokens_list, true_labels_list, predictions, args):
        """生成对比真实标签与预测标签的评估式报告
        
        Args:
            texts: 原始文本列表
            tokens_list: tokens列表
            true_labels_list: 真实标签列表
            predictions: 预测结果列表
            args: 命令行参数
        """
        try:
            from datetime import datetime
            from pathlib import Path
            from ..evaluation.report_generator import NERReportGenerator
            
            self.logger.info("Starting to generate comparison report...")
            
            # 确定输出目录
            output_dir = None
            if getattr(args, 'output_file', None):
                output_dir = Path(args.output_file).parent
            else:
                output_dir = Path.cwd()
            
            # 从真实标签中提取实体类型
            entity_types = self._extract_entity_types_from_labels(true_labels_list)
            
            if not entity_types:
                self.logger.warning("No entity types found in true labels, falling back to prediction report")
                self._generate_prediction_report(texts, predictions, args)
                return
            
            self.logger.info(f"Detected entity types: {entity_types}")
            
            # 提取预测的标签序列
            pred_labels_list = [pred.get('labels', []) for pred in predictions]
            
            # 生成报告文件路径
            timestamp = datetime.now().strftime("%Y%m%d%H%M")
            report_filename = f"prediction_comparison_report_{timestamp}.xlsx"
            report_path = output_dir / report_filename
            
            # 构造评估结果字典（简化版，主要用于报告生成）
            evaluation_results = {
                'num_samples': len(texts),
                'token_metrics': {},
                'entity_metrics': {},
                'per_entity_metrics': {},
                'predictions': pred_labels_list
            }
            
            # 使用NERReportGenerator生成Excel报告
            report_generator = NERReportGenerator(logger=self.logger)
            report_path = report_generator.generate_excel_report(
                texts=texts,
                true_labels=true_labels_list,
                predictions=pred_labels_list,
                evaluation_results=evaluation_results,
                entity_types=entity_types,
                output_path=str(report_path),
                specified_entities=None,
                tokens_list=tokens_list
            )
            
            if report_path:
                self.logger.info(f"Comparison report generated: {report_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate comparison report: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
    
    def _generate_prediction_report(self, texts, predictions, args):
        """生成预测结果的详细报告
        
        Args:
            texts: 原始文本列表
            predictions: 预测结果列表，每个元素包含 'tokens' 和 'labels'
            args: 命令行参数
        """
        try:
            from datetime import datetime
            from pathlib import Path
            
            self.logger.info("Starting to generate prediction report...")
            
            # 确定输出目录
            output_dir = None
            if getattr(args, 'output_file', None):
                output_dir = Path(args.output_file).parent
            else:
                output_dir = Path.cwd()
            
            # 从预测结果中提取实体类型
            entity_types = self._extract_entity_types_from_predictions(predictions)
            
            if not entity_types:
                self.logger.warning("No entity types found in predictions, skipping detailed report")
                return
            
            self.logger.info(f"Detected entity types: {entity_types}")
            
            # 生成报告文件路径
            timestamp = datetime.now().strftime("%Y%m%d%H%M")
            report_filename = f"prediction_report_{timestamp}.xlsx"
            report_path = output_dir / report_filename
            
            # 生成 Excel 报告
            self._generate_prediction_excel(texts, predictions, entity_types, report_path)
            
            self.logger.info(f"Prediction report generated: {report_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate prediction report: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
    
    def _extract_entity_types_from_labels(self, labels_list):
        """从标签列表中提取所有实体类型
        
        Args:
            labels_list: 标签序列列表
            
        Returns:
            实体类型列表（排序后的去重列表）
        """
        entity_types = set()
        
        for labels in labels_list:
            for label in labels:
                if label != 'O' and '-' in label:
                    entity_type = label.split('-', 1)[1]
                    entity_types.add(entity_type)
        
        return sorted(list(entity_types))
    
    def _extract_entity_types_from_predictions(self, predictions):
        """从预测结果中提取所有实体类型
        
        Args:
            predictions: 预测结果列表
            
        Returns:
            实体类型列表（排序后的去重列表）
        """
        entity_types = set()
        
        for pred in predictions:
            labels = pred.get('labels', [])
            for label in labels:
                if label != 'O' and '-' in label:
                    entity_type = label.split('-', 1)[1]
                    entity_types.add(entity_type)
        
        return sorted(list(entity_types))
    
    def _generate_prediction_excel(self, texts, predictions, entity_types, report_path):
        """生成预测结果的 Excel 报告
        
        Args:
            texts: 原始文本列表
            predictions: 预测结果列表
            entity_types: 实体类型列表
            report_path: 报告输出路径
        """
        try:
            import openpyxl
            from openpyxl.styles import PatternFill, Font, Alignment
        except ImportError:
            self.logger.error("openpyxl is required for Excel report generation. Install with: pip install openpyxl")
            return
        
        # 创建输出目录
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 创建工作簿
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Prediction Report"
        
        # 写入表头
        headers = ['ADDRESS'] + entity_types
        for col, header in enumerate(headers, 1):
            cell = ws.cell(row=1, column=col, value=header)
            # 设置表头样式
            cell.fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
            cell.font = Font(bold=True, color="FFFFFF")
            cell.alignment = Alignment(horizontal="center", vertical="center")
        
        # 写入数据行
        for idx, (text, pred) in enumerate(zip(texts, predictions), start=2):
            # ADDRESS 列
            ws.cell(row=idx, column=1, value=text)
            
            # 提取实体
            tokens = pred.get('tokens', [])
            labels = pred.get('labels', [])
            entities = self._extract_entities_from_labels(tokens, labels)
            
            # 为每个实体类型填充数据
            for col, entity_type in enumerate(entity_types, start=2):
                entity_text = entities.get(entity_type, "")
                cell = ws.cell(row=idx, column=col, value=entity_text)
                
                # 如果识别出实体，高亮显示
                if entity_text:
                    cell.fill = PatternFill(start_color="C6E0B4", end_color="C6E0B4", fill_type="solid")
        
        # 自动调整列宽
        self._auto_adjust_column_width(ws)
        
        # 保存文件
        wb.save(report_path)
        self.logger.info(f"Excel report saved to: {report_path}")
    
    def _extract_entities_from_labels(self, tokens, labels):
        """从标签序列中提取实体
        
        Args:
            tokens: 词序列
            labels: 标签序列
            
        Returns:
            实体类型到实体文本的映射（同一类型多个实体用 " | " 连接）
        """
        entities = {}
        current_entity = None
        current_tokens = []
        
        for token, label in zip(tokens, labels):
            if label.startswith('B-'):
                # 开始新实体
                if current_entity:
                    # 保存之前的实体
                    entity_text = ' '.join(current_tokens)
                    if current_entity in entities:
                        entities[current_entity] += f" | {entity_text}"
                    else:
                        entities[current_entity] = entity_text
                current_entity = label[2:]
                current_tokens = [token]
            elif label.startswith('I-'):
                # 继续当前实体
                entity_type = label[2:]
                if current_entity == entity_type:
                    current_tokens.append(token)
                else:
                    # 不匹配的I-标签，先保存当前实体，然后将此I-视为新的B-
                    if current_entity:
                        entity_text = ' '.join(current_tokens)
                        if current_entity in entities:
                            entities[current_entity] += f" | {entity_text}"
                        else:
                            entities[current_entity] = entity_text
                    current_entity = entity_type
                    current_tokens = [token]
            else:
                # O标签或其他：结束当前实体
                if current_entity:
                    entity_text = ' '.join(current_tokens)
                    if current_entity in entities:
                        entities[current_entity] += f" | {entity_text}"
                    else:
                        entities[current_entity] = entity_text
                    current_entity = None
                    current_tokens = []
        
        # 处理最后一个实体
        if current_entity:
            entity_text = ' '.join(current_tokens)
            if current_entity in entities:
                entities[current_entity] += f" | {entity_text}"
            else:
                entities[current_entity] = entity_text
        
        return entities
    
    def _auto_adjust_column_width(self, ws):
        """自动调整列宽以适应内容"""
        try:
            for column in ws.columns:
                max_length = 0
                column_letter = column[0].column_letter
                
                for cell in column:
                    try:
                        if cell.value:
                            # 考虑中文字符（占2个单位）和英文字符（占1个单位）
                            cell_len = sum(2 if ord(c) > 127 else 1 for c in str(cell.value))
                            max_length = max(max_length, cell_len)
                    except (TypeError, ValueError, AttributeError):
                        pass
                
                # 设置列宽，增加一些余量，并设置最大和最小值
                adjusted_width = min(max(max_length + 2, 10), 100)
                ws.column_dimensions[column_letter].width = adjusted_width
        except Exception:
            # 如果自动调整失败，静默跳过
            pass
