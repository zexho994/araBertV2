"""NER 评估报告生成器

提供详细的评估报告生成功能，包括：
- CSV 格式的详细评估报告
- 实体识别错误的高亮显示
- 汇总统计信息
- 逐行识别结果分析
"""

import csv
from typing import Dict, List, Any, Optional
from pathlib import Path
from ..utils import NERLogger


class NERReportGenerator:
    """NER 评估报告生成器
    
    职责：
    - 生成包含汇总信息和详细识别结果的 CSV 报告
    - 支持实体识别错误的高亮显示
    - 提供多种报告格式和选项
    """
    
    def __init__(self, logger: Optional[NERLogger] = None):
        self.logger = logger or NERLogger()
    
    def _prepare_report_data(
        self,
        texts: List[str],
        true_labels: List[List[str]],
        predictions: List[List[str]],
        evaluation_results: Dict[str, Any],
        entity_types: List[str],
        specified_entities: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """准备报告数据
        
        Returns:
            包含汇总信息和详细结果的数据字典
        """
        # 提取汇总信息
        summary = self._extract_summary(evaluation_results, entity_types)
        
        # 生成逐行详细结果
        detailed_results = self._generate_detailed_results(
            texts, true_labels, predictions, entity_types
        )
        
        result = {
            'summary': summary,
            'detailed_results': detailed_results
        }
        
        # 如果指定了实体类型，生成问题实体数据
        if specified_entities:
            problematic_entities = self._generate_problematic_entities_data(
                texts, true_labels, predictions, specified_entities
            )
            result['problematic_entities'] = problematic_entities
        
        return result
    
    def _extract_summary(
        self,
        evaluation_results: Dict[str, Any],
        entity_types: List[str]
    ) -> Dict[str, Any]:
        """提取汇总信息"""
        summary = {
            'evaluation_summary': {
                'total_samples': evaluation_results.get('num_samples', 0),
                'token_metrics': evaluation_results.get('token_metrics', {}),
                'entity_metrics': evaluation_results.get('entity_metrics', {}),
                'per_entity_metrics': evaluation_results.get('per_entity_metrics', {})
            },
            'entity_types': entity_types,
            'timestamp': self._get_timestamp()
        }
        return summary
    
    def _generate_detailed_results(
        self,
        texts: List[str],
        true_labels: List[List[str]],
        predictions: List[List[str]],
        entity_types: List[str]
    ) -> List[Dict[str, Any]]:
        """生成逐行详细结果"""
        
        detailed_results = []
        
        for text, true_seq, pred_seq in zip(texts, true_labels, predictions):
            # 提取实体信息
            true_entities = self._extract_entities_from_sequence(true_seq, text.split())
            pred_entities = self._extract_entities_from_sequence(pred_seq, text.split())
            
            # 生成每行的结果
            row_result = {
                'sample_id': len(detailed_results) + 1,
                'text': text,
                'tokens': text.split(),
                'true_labels': true_seq,
                'pred_labels': pred_seq,
                'true_entities': true_entities,
                'pred_entities': pred_entities,
                'entity_columns': {}
            }
            
            # 为每个实体类型生成列
            for entity_type in entity_types:
                true_entity_text = self._get_entity_text(true_entities, entity_type)
                pred_entity_text = self._get_entity_text(pred_entities, entity_type)
                
                # 检查是否有识别错误
                if true_entity_text != pred_entity_text:
                    # 有差异，使用不同格式展示
                    if true_entity_text and pred_entity_text:
                        # 都有值但不同：真正的误识别
                        cell_content = f"正确: {true_entity_text} | 预测: {pred_entity_text}"
                        cell_style = "error"
                    elif true_entity_text and not pred_entity_text:
                        # 标注有值但未识别：漏识别 - 使用灰色
                        cell_content = f"正确: {true_entity_text}"
                        cell_style = "missing"  # 使用missing样式（灰色）
                    else:
                        # 标注无值但预测有值：可能是标注缺失或真的误识别
                        # 使用中性的展示方式，让用户自行判断
                        cell_content = f"预测: {pred_entity_text}"
                        cell_style = "warning"  # 使用warning样式（黄色）
                else:
                    # 完全一致
                    if true_entity_text:
                        # 正确识别
                        cell_content = true_entity_text
                        cell_style = "correct"
                    else:
                        # 都为空
                        cell_content = ""
                        cell_style = "correct"
                
                row_result['entity_columns'][entity_type] = {
                    'content': cell_content,
                    'style': cell_style,
                    'true': true_entity_text,
                    'pred': pred_entity_text
                }
            
            detailed_results.append(row_result)
        
        return detailed_results
    
    def _extract_entities_from_sequence(
        self, 
        labels: List[str], 
        tokens: List[str]
    ) -> Dict[str, str]:
        """从标签序列中提取实体文本
        
        Args:
            labels: 标签序列
            tokens: 对应的词序列
            
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
                        # 如果该类型已存在，追加
                        entities[current_entity] += f" | {entity_text}"
                    else:
                        entities[current_entity] = entity_text
                current_entity = label[2:]  # 去掉 'B-' 前缀
                current_tokens = [token]
            elif label.startswith('I-') and current_entity == label[2:]:
                # 继续当前实体
                current_tokens.append(token)
            else:
                # 结束当前实体
                if current_entity:
                    entity_text = ' '.join(current_tokens)
                    if current_entity in entities:
                        # 如果该类型已存在，追加
                        entities[current_entity] += f" | {entity_text}"
                    else:
                        entities[current_entity] = entity_text
                    current_entity = None
                    current_tokens = []
        
        # 处理最后一个实体
        if current_entity:
            entity_text = ' '.join(current_tokens)
            if current_entity in entities:
                # 如果该类型已存在，追加
                entities[current_entity] += f" | {entity_text}"
            else:
                entities[current_entity] = entity_text
        
        return entities
    
    def _get_entity_text(self, entities: Dict[str, str], entity_type: str) -> str:
        """获取指定实体类型的文本"""
        return entities.get(entity_type, "")
    
    def _write_csv_report(self, report_data: Dict[str, Any], output_path: Path, specified_entities: Optional[List[str]] = None):
        """写入 CSV 报告文件"""
        summary = report_data['summary']
        detailed_results = report_data['detailed_results']
        entity_types = summary['entity_types']
        
        # 创建 CSV 文件
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # 移除汇总信息，直接写入详细结果
            # self._write_summary_section(writer, summary)  # 已注释
            
            # 写入详细结果
            self._write_detailed_section(writer, detailed_results, entity_types)
            
            # 如果指定了实体类型，写入问题实体部分
            if specified_entities and 'problematic_entities' in report_data:
                self._write_csv_problematic_entities(writer, report_data['problematic_entities'], specified_entities)
    
    def _write_summary_section(self, writer, summary: Dict[str, Any]):
        """写入汇总信息部分"""
        writer.writerow(['=== EVALUATION SUMMARY ==='])
        writer.writerow([])
        
        # 基本信息
        eval_summary = summary['evaluation_summary']
        writer.writerow(['Total Samples', eval_summary['total_samples']])
        writer.writerow(['Timestamp', summary['timestamp']])
        writer.writerow([])
        
        # Token 级指标
        writer.writerow(['=== TOKEN-LEVEL METRICS ==='])
        token_metrics = eval_summary['token_metrics']
        for metric, value in token_metrics.items():
            writer.writerow([f'Token {metric.title()}', f'{value:.4f}'])
        writer.writerow([])
        
        # 实体级指标
        writer.writerow(['=== ENTITY-LEVEL METRICS ==='])
        entity_metrics = eval_summary['entity_metrics']
        for metric, value in entity_metrics.items():
            writer.writerow([f'Entity {metric.title()}', f'{value:.4f}'])
        writer.writerow([])
        
        # 逐实体指标
        writer.writerow(['=== PER-ENTITY METRICS ==='])
        per_entity_metrics = eval_summary['per_entity_metrics']
        for entity, metrics in per_entity_metrics.items():
            writer.writerow([f'Entity: {entity}'])
            for metric, value in metrics.items():
                writer.writerow([f'  {metric.title()}', f'{value:.4f}'])
        writer.writerow([])
        
        # 分隔线
        writer.writerow(['=== DETAILED RESULTS ==='])
        writer.writerow([])
    
    def _write_detailed_section(
        self, 
        writer, 
        detailed_results: List[Dict[str, Any]], 
        entity_types: List[str]
    ):
        """写入详细结果部分"""
        # 写入表头
        # 确保 entity_types 是列表
        if isinstance(entity_types, dict):
            entity_types = list(entity_types.keys())
        elif not isinstance(entity_types, list):
            entity_types = list(entity_types)
        
        # 移除 Sample ID 列，Text 改名为 ADDRESS
        headers = ['ADDRESS'] + entity_types
        writer.writerow(headers)
        
        # 写入每行数据
        for result in detailed_results:
            row = [
                result['text']
            ]
            
            # 添加每个实体类型的列
            for entity_type in entity_types:
                entity_data = result['entity_columns'][entity_type]
                row.append(entity_data['content'])
            
            writer.writerow(row)
    
    def _write_csv_problematic_entities(
        self, 
        writer, 
        problematic_entities: List[Dict[str, Any]], 
        specified_entities: List[str]
    ):
        """写入CSV问题实体部分"""
        if not problematic_entities:
            return
        
        # 写入标题
        writer.writerow([])
        writer.writerow(['=== PROBLEMATIC ENTITIES SECTION ==='])
        writer.writerow([f"Samples where all specified entities ({', '.join(specified_entities)}) have issues"])
        writer.writerow([])
        
        # 写入表头（移除 Sample ID，Text 改名为 ADDRESS）
        headers = ['ADDRESS'] + [f"{entity} (正确值)" for entity in specified_entities] + [f"{entity} (预测)" for entity in specified_entities]
        writer.writerow(headers)
        
        # 写入数据行
        for sample in problematic_entities:
            row = [
                sample['text']
            ]
            
            # 添加实体列
            for entity_type in specified_entities:
                entity_issue = sample['entity_issues'][entity_type]
                row.append(entity_issue['true'] or "")
                row.append(entity_issue['pred'] or "")
            
            writer.writerow(row)
    
    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    def generate_excel_report(
        self,
        texts: List[str],
        true_labels: List[List[str]],
        predictions: List[List[str]],
        evaluation_results: Dict[str, Any],
        entity_types: List[str],
        output_path: str,
        specified_entities: Optional[List[str]] = None
    ) -> str:
        """生成 Excel 格式的评估报告（带颜色高亮）
        
        Args:
            texts: 原始文本列表
            true_labels: 真实标签序列
            predictions: 预测标签序列
            evaluation_results: 评估结果字典
            entity_types: 实体类型列表
            output_path: 输出文件路径
            specified_entities: 指定的实体类型列表，如果提供，将创建特殊的问题实体部分
            
        Returns:
            生成的报告文件路径
        """
        try:
            import openpyxl
        except ImportError:
            self.logger.error("openpyxl is required for Excel report generation. Install with: pip install openpyxl")
            return ""
        
        self.logger.info(f"Generating Excel evaluation report to: {output_path}")
        
        # 创建输出目录
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 生成报告数据
        report_data = self._prepare_report_data(
            texts, true_labels, predictions, evaluation_results, entity_types, specified_entities
        )
        
        # 创建 Excel 工作簿
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Evaluation Report"
        
        # 写入汇总信息
        self._write_excel_summary(ws, report_data['summary'])
        
        # 写入详细结果
        self._write_excel_detailed(ws, report_data['detailed_results'], entity_types)
        
        # 如果指定了实体类型，写入问题实体部分
        if specified_entities and 'problematic_entities' in report_data:
            self._write_excel_problematic_entities(ws, report_data['problematic_entities'], specified_entities)
        
        # 保存文件
        wb.save(output_path)
        
        # 如果指定了实体类型且有问题数据，生成JSONL文件
        if specified_entities and 'problematic_entities' in report_data and report_data['problematic_entities']:
            jsonl_path = self._generate_problematic_entities_jsonl(
                report_data['problematic_entities'], 
                str(output_path)
            )
            if jsonl_path:
                self.logger.info(f"Problematic entities JSONL saved to: {jsonl_path}")
        
        self.logger.info(f"Excel evaluation report saved to: {output_path}")
        return str(output_path)
    
    def _write_excel_summary(self, ws, summary: Dict[str, Any]):
        """写入 Excel 汇总信息（已简化，仅返回起始行）"""
        # 直接从第一行开始，不写入任何汇总信息
        return 1
    
    def _write_excel_detailed(self, ws, detailed_results: List[Dict[str, Any]], entity_types: List[str]):
        """写入 Excel 详细结果"""
        # 直接从第一行开始写入表头（已移除汇总部分）
        summary_end_row = 1
        
        # 写入表头
        # 确保 entity_types 是列表
        if isinstance(entity_types, dict):
            entity_types = list(entity_types.keys())
        elif not isinstance(entity_types, list):
            entity_types = list(entity_types)
        
        # 移除 Sample ID 列，Text 改名为 ADDRESS
        headers = ['ADDRESS'] + entity_types
        for col, header in enumerate(headers, 1):
            ws.cell(row=summary_end_row, column=col, value=header)
        
        # 设置表头样式
        try:
            from openpyxl.styles import PatternFill
            header_fill = PatternFill(start_color="CCCCCC", end_color="CCCCCC", fill_type="solid")
            for col in range(1, len(headers) + 1):
                ws.cell(row=summary_end_row, column=col).fill = header_fill
        except ImportError:
            pass  # 如果 openpyxl 不可用，跳过样式设置
        
        # 写入数据行
        data_start_row = summary_end_row + 1
        try:
            from openpyxl.styles import PatternFill
            error_fill = PatternFill(start_color="FFCCCC", end_color="FFCCCC", fill_type="solid")  # 红色：真正的错误
            missing_fill = PatternFill(start_color="D3D3D3", end_color="D3D3D3", fill_type="solid")  # 灰色：漏识别
            warning_fill = PatternFill(start_color="FFE699", end_color="FFE699", fill_type="solid")  # 黄色：可能是标注缺失
        except ImportError:
            error_fill = None
            missing_fill = None
            warning_fill = None
        
        for idx, result in enumerate(detailed_results):
            row = data_start_row + idx
            
           # ADDRESS 列（移除了 Sample ID）
            ws.cell(row=row, column=1, value=result['text'])
            
            # 实体列（列索引从 2 开始）
            for j, entity_type in enumerate(entity_types):
                col = 2 + j
                entity_data = result['entity_columns'][entity_type]
                cell = ws.cell(row=row, column=col, value=entity_data['content'])
                
                # 根据样式设置背景色
                if entity_data['style'] == 'error' and error_fill is not None:
                    cell.fill = error_fill
                elif entity_data['style'] == 'missing' and missing_fill is not None:
                    cell.fill = missing_fill
                elif entity_data['style'] == 'warning' and warning_fill is not None:
                    cell.fill = warning_fill
        
        # 自动调整列宽
        self._auto_adjust_column_width(ws)
    
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
    
    def _generate_problematic_entities_data(
        self,
        texts: List[str],
        true_labels: List[List[str]],
        predictions: List[List[str]],
        specified_entities: List[str]
    ) -> List[Dict[str, Any]]:
        """生成问题实体数据
        
        找出所有指定实体类型都有问题的样本
        
        Args:
            texts: 原始文本列表
            true_labels: 真实标签序列
            predictions: 预测标签序列
            specified_entities: 指定的实体类型列表
            
        Returns:
            问题实体数据列表
        """
        problematic_samples = []
        
        for i, (text, true_seq, pred_seq) in enumerate(zip(texts, true_labels, predictions)):
            # 提取实体信息
            true_entities = self._extract_entities_from_sequence(true_seq, text.split())
            pred_entities = self._extract_entities_from_sequence(pred_seq, text.split())
            
            # 检查指定实体类型是否都有问题
            all_entities_problematic = True
            entity_issues = {}
            
            for entity_type in specified_entities:
                true_entity_text = self._get_entity_text(true_entities, entity_type)
                pred_entity_text = self._get_entity_text(pred_entities, entity_type)
                
                # 检查是否有识别错误
                has_error = true_entity_text != pred_entity_text
                entity_issues[entity_type] = {
                    'true': true_entity_text,
                    'pred': pred_entity_text,
                    'has_error': has_error
                }
                
                if not has_error:
                    all_entities_problematic = False
                    break
            
            # 如果所有指定实体都有问题，添加到问题样本列表
            if all_entities_problematic and any(issue['has_error'] for issue in entity_issues.values()):
                problematic_samples.append({
                    'sample_id': i + 1,
                    'text': text,
                    'tokens': text.split(),
                    'true_labels': true_seq,
                    'pred_labels': pred_seq,
                    'entity_issues': entity_issues
                })
        
        return problematic_samples
    
    def _write_excel_problematic_entities(
        self, 
        ws, 
        problematic_entities: List[Dict[str, Any]], 
        specified_entities: List[str]
    ):
        """写入Excel问题实体部分"""
        if not problematic_entities:
            return
        
        # 找到当前内容的结束行
        current_row = ws.max_row + 2
        
        # 写入标题
        ws.cell(row=current_row, column=1, value="=== PROBLEMATIC ENTITIES SECTION ===")
        current_row += 1
        ws.cell(row=current_row, column=1, value=f"Samples where all specified entities ({', '.join(specified_entities)}) have issues")
        current_row += 2
        
        # 写入表头（移除 Sample ID，Text 改名为 ADDRESS）
        headers = ['ADDRESS'] + [f"{entity} (正确)" for entity in specified_entities] + [f"{entity} (预测)" for entity in specified_entities]
        for col, header in enumerate(headers, 1):
            ws.cell(row=current_row, column=col, value=header)
        
        # 设置表头样式
        try:
            from openpyxl.styles import PatternFill
            header_fill = PatternFill(start_color="FF9999", end_color="FF9999", fill_type="solid")
            for col in range(1, len(headers) + 1):
                ws.cell(row=current_row, column=col).fill = header_fill
        except ImportError:
            pass
        
        current_row += 1
        
        # 写入数据行
        try:
            from openpyxl.styles import PatternFill
            error_fill = PatternFill(start_color="FFCCCC", end_color="FFCCCC", fill_type="solid")
        except ImportError:
            error_fill = None
        
        for sample in problematic_entities:
            # ADDRESS 列（移除了 Sample ID）
            ws.cell(row=current_row, column=1, value=sample['text'])
            
            # 实体列（列索引从 2 开始）
            col = 2
            for entity_type in specified_entities:
                entity_issue = sample['entity_issues'][entity_type]
                
                # 正确
                ws.cell(row=current_row, column=col, value=entity_issue['true'] or "")
                if error_fill is not None:
                    ws.cell(row=current_row, column=col).fill = error_fill
                col += 1
                
                # 预测值
                ws.cell(row=current_row, column=col, value=entity_issue['pred'] or "")
                if error_fill is not None:
                    ws.cell(row=current_row, column=col).fill = error_fill
                col += 1
            
            current_row += 1
    
    def _generate_problematic_entities_jsonl(
        self, 
        problematic_entities: List[Dict[str, Any]], 
        excel_path: str
    ) -> Optional[str]:
        """生成问题实体的JSONL文件
        
        Args:
            problematic_entities: 问题实体数据列表
            excel_path: Excel文件路径，用于生成对应的JSONL文件名
            
        Returns:
            生成的JSONL文件路径，如果失败返回None
        """
        if not problematic_entities:
            return None
        
        try:
            import json
            from pathlib import Path
            
            # 基于Excel文件路径生成JSONL文件路径
            excel_path = Path(excel_path)
            jsonl_filename = excel_path.stem + "_problematic_entities.jsonl"
            jsonl_path = excel_path.parent / jsonl_filename
            
            self.logger.info(f"Generating problematic entities JSONL to: {jsonl_path}")
            
            # 写入JSONL文件
            with open(jsonl_path, 'w', encoding='utf-8') as f:
                for sample in problematic_entities:
                    # 构建JSONL记录，保持与输入JSONL相同的格式
                    record = {
                        'text': sample['text'],
                        'tokens': sample['tokens'],
                        'labels': sample['true_labels']
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            self.logger.info(f"Problematic entities JSONL saved: {jsonl_path}")
            return str(jsonl_path)
            
        except Exception as e:
            self.logger.error(f"Failed to generate problematic entities JSONL: {e}")
            return None
