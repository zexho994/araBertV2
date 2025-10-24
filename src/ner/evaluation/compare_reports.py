"""NER 模型评估报告对比工具

提供两个模型评估报告的对比功能，包括：
- 加载 XLSX 格式的评估报告
- 按地址匹配两个模型的预测结果
- 生成垂直堆叠格式的对比报告
- 用颜色高亮标记预测差异
"""

from typing import Optional, List, Dict, Any
from pathlib import Path
import pandas as pd
import openpyxl
from openpyxl.styles import PatternFill, Font, Alignment
from ..utils import NERLogger


class ReportComparator:
    """NER 评估报告对比器
    
    职责：
    - 读取两个模型的评估报告
    - 按地址匹配并合并报告
    - 生成带颜色标记的对比报告
    """
    
    # 颜色定义
    COLOR_SAME = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")  # 绿色：预测一致
    COLOR_DIFF = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")  # 黄色：预测不同
    COLOR_HEADER = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")  # 蓝色：表头
    COLOR_MODEL_NAME = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")  # 浅蓝色：模型名称列
    
    def __init__(self, logger: Optional[NERLogger] = None):
        self.logger = logger or NERLogger()
    
    def load_report(self, report_path: str) -> pd.DataFrame:
        """加载评估报告文件
        
        Args:
            report_path: 报告文件路径（支持 xlsx 和 csv 格式）
            
        Returns:
            pandas DataFrame
        """
        report_path = Path(report_path)
        
        if not report_path.exists():
            raise FileNotFoundError(f"Report file not found: {report_path}")
        
        self.logger.info(f"Loading report from: {report_path}")
        
        # 读取 Excel 文件
        if report_path.suffix.lower() in ['.xlsx', '.xls']:
            # 新格式：直接从第一行开始就是表头（ADDRESS）
            # 不再需要查找 "=== DETAILED RESULTS ===" 标记
            df = pd.read_excel(report_path, header=0)
            
            # 删除完全为空的行
            df = df.dropna(how='all')
            
            self.logger.info(f"Loaded {len(df)} samples from report")
            return df
            
        elif report_path.suffix.lower() == '.csv':
            # 读取 CSV 文件
            # 新格式：直接从第一行开始就是表头（ADDRESS）
            df = pd.read_csv(report_path, sep=',', skipinitialspace=True)
            
            # 删除完全为空的行
            df = df.dropna(how='all')
            
            # 清理列名（去除可能的制表符等）
            df.columns = df.columns.str.strip()
            
            self.logger.info(f"Loaded {len(df)} samples from CSV report")
            return df
            
        else:
            raise ValueError(f"Unsupported file format: {report_path.suffix}. Only .xlsx and .csv are supported.")
    
    def extract_entity_value(self, cell_content: str) -> str:
        """从单元格内容中提取实际的实体值
        
        处理格式：
        - "预测: value" -> "value"
        - "正确: value | 预测: value2" -> "value2" (提取预测值)
        - "正确: value" -> "value" (只有正确值，说明预测为空)
        - "" -> ""
        - "value" -> "value" (直接是值)
        
        Args:
            cell_content: 单元格内容
            
        Returns:
            提取的实体值
        """
        if pd.isna(cell_content) or cell_content == "":
            return ""
        
        cell_content = str(cell_content).strip()
        
        # 如果包含 "预测:"，提取预测值
        if "预测:" in cell_content or "预测：" in cell_content:
            # 处理 "预测: value" 或 "正确: xxx | 预测: value" 格式
            parts = cell_content.split("|")
            for part in parts:
                part = part.strip()
                if part.startswith("预测:") or part.startswith("预测："):
                    # 提取 "预测:" 后面的内容
                    value = part.split(":", 1)[1].strip() if ":" in part else part.split("：", 1)[1].strip()
                    return value
        
        # 如果只有 "正确:" 没有 "预测:"，说明预测为空
        if "正确:" in cell_content or "正确：" in cell_content:
            return ""
        
        # 否则直接返回内容（可能是正确识别的值）
        return cell_content
    
    def merge_reports(
        self,
        report1: pd.DataFrame,
        report2: pd.DataFrame,
        model1_name: str = "Model-1",
        model2_name: str = "Model-2"
    ) -> List[Dict[str, Any]]:
        """合并两个报告，生成垂直堆叠格式
        
        Args:
            report1: 第一个模型的报告
            report2: 第二个模型的报告
            model1_name: 第一个模型的名称
            model2_name: 第二个模型的名称
            
        Returns:
            合并后的数据列表，每个元素包含一个地址的两个模型预测
        """
        self.logger.info(f"Merging reports: {model1_name} vs {model2_name}")
        
        # 检查两个报告是否有相同的列结构
        if not report1.columns.equals(report2.columns):
            # 尝试对齐列
            all_columns = list(report1.columns.union(report2.columns))
            report1 = report1.reindex(columns=all_columns, fill_value="")
            report2 = report2.reindex(columns=all_columns, fill_value="")
            self.logger.warning("Reports have different columns, aligned automatically")
        
        # 第一列应该是 ADDRESS（新格式已统一命名）
        address_col = report1.columns[0]
        
        # 验证第一列是否为 ADDRESS
        if address_col.upper() not in ['ADDRESS', 'TEXT']:
            self.logger.warning(f"First column is '{address_col}', expected 'ADDRESS' or 'TEXT'")
        
        entity_columns = report1.columns[1:].tolist()  # 其余列是实体类型
        
        merged_data = []
        
        # 按地址匹配
        for _, row1 in report1.iterrows():
            address = row1[address_col]
            
            # 在 report2 中查找相同的地址
            matching_rows = report2[report2[address_col] == address]
            
            if matching_rows.empty:
                self.logger.warning(f"Address not found in report2: {address}")
                continue
            
            row2 = matching_rows.iloc[0]
            
            # 保存原始内容和提取的值（用于比较）
            model1_entities_raw = {}      # 原始内容
            model2_entities_raw = {}      # 原始内容
            model1_entities_extracted = {}  # 提取的值（用于比较）
            model2_entities_extracted = {}  # 提取的值（用于比较）
            
            for entity_col in entity_columns:
                # 保存原始内容
                model1_entities_raw[entity_col] = row1[entity_col] if not pd.isna(row1[entity_col]) else ""
                model2_entities_raw[entity_col] = row2[entity_col] if not pd.isna(row2[entity_col]) else ""
                
                # 提取值用于比较
                model1_entities_extracted[entity_col] = self.extract_entity_value(row1[entity_col])
                model2_entities_extracted[entity_col] = self.extract_entity_value(row2[entity_col])
            
            # 比较实体，标记差异（使用提取的值比较，但显示原始内容）
            entity_comparison = {}
            for entity_col in entity_columns:
                val1_extracted = model1_entities_extracted[entity_col]
                val2_extracted = model2_entities_extracted[entity_col]
                is_same = val1_extracted == val2_extracted
                
                entity_comparison[entity_col] = {
                    'model1_value': model1_entities_raw[entity_col],  # 使用原始内容
                    'model2_value': model2_entities_raw[entity_col],  # 使用原始内容
                    'is_same': is_same  # 但使用提取的值来判断是否相同
                }
            
            merged_data.append({
                'address': address,
                'model1_name': model1_name,
                'model2_name': model2_name,
                'entity_columns': entity_columns,
                'entity_comparison': entity_comparison
            })
        
        self.logger.info(f"Merged {len(merged_data)} addresses")
        return merged_data
    
    def generate_comparison_report(
        self,
        report1_path: str,
        report2_path: str,
        output_path: str,
        model1_name: str = "Model-1",
        model2_name: str = "Model-2"
    ) -> str:
        """生成对比报告
        
        Args:
            report1_path: 第一个模型的报告路径
            report2_path: 第二个模型的报告路径
            output_path: 输出文件路径
            model1_name: 第一个模型名称
            model2_name: 第二个模型名称
            
        Returns:
            生成的报告文件路径
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting report comparison")
        self.logger.info("=" * 60)
        
        # 加载报告
        report1 = self.load_report(report1_path)
        report2 = self.load_report(report2_path)
        
        # 合并报告
        merged_data = self.merge_reports(report1, report2, model1_name, model2_name)
        
        # 生成 Excel 文件
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._write_excel_comparison(merged_data, output_path)
        
        self.logger.info("=" * 60)
        self.logger.info(f"Comparison report saved to: {output_path}")
        self.logger.info("=" * 60)
        
        return str(output_path)
    
    def _write_excel_comparison(self, merged_data: List[Dict[str, Any]], output_path: Path):
        """写入 Excel 对比报告
        
        Args:
            merged_data: 合并后的数据
            output_path: 输出文件路径
        """
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Model Comparison"
        
        # 确定表头
        if not merged_data:
            self.logger.warning("No data to write")
            wb.save(output_path)
            return
        
        entity_columns = merged_data[0]['entity_columns']
        headers = ['ADDRESS', 'MODEL_NAME'] + entity_columns
        
        # 从第一行开始写入表头
        header_row = 1
        
        # 写入表头
        for col_idx, header in enumerate(headers, start=1):
            cell = ws.cell(row=header_row, column=col_idx, value=header)
            cell.fill = self.COLOR_HEADER
            cell.font = Font(bold=True, color="FFFFFF")
            cell.alignment = Alignment(horizontal='center', vertical='center')
        
        # 写入数据
        current_row = header_row + 1
        
        for data in merged_data:
            address = data['address']
            model1_name = data['model1_name']
            model2_name = data['model2_name']
            entity_comparison = data['entity_comparison']
            
            # 记录起始行用于合并单元格
            address_start_row = current_row
            
            # 写入 Model 1 的行
            address_cell = ws.cell(row=current_row, column=1, value=address)
            address_cell.alignment = Alignment(vertical='center')  # 垂直居中
            
            model_cell = ws.cell(row=current_row, column=2, value=model1_name)
            model_cell.fill = self.COLOR_MODEL_NAME
            model_cell.font = Font(bold=True)
            
            for col_idx, entity_col in enumerate(entity_columns, start=3):
                comp = entity_comparison[entity_col]
                cell = ws.cell(row=current_row, column=col_idx, value=comp['model1_value'])
                
                # 根据是否一致设置颜色
                if comp['is_same']:
                    cell.fill = self.COLOR_SAME
                else:
                    cell.fill = self.COLOR_DIFF
            
            current_row += 1
            
            # 写入 Model 2 的行
            ws.cell(row=current_row, column=1, value=address)  # 先写入值，后面会被合并
            model_cell = ws.cell(row=current_row, column=2, value=model2_name)
            model_cell.fill = self.COLOR_MODEL_NAME
            model_cell.font = Font(bold=True)
            
            for col_idx, entity_col in enumerate(entity_columns, start=3):
                comp = entity_comparison[entity_col]
                cell = ws.cell(row=current_row, column=col_idx, value=comp['model2_value'])
                
                # 根据是否一致设置颜色
                if comp['is_same']:
                    cell.fill = self.COLOR_SAME
                else:
                    cell.fill = self.COLOR_DIFF
            
            # 合并 ADDRESS 单元格（两行）
            ws.merge_cells(start_row=address_start_row, start_column=1, 
                          end_row=current_row, end_column=1)
            
            current_row += 1
            
            # 插入空行分隔
            current_row += 1
        
        # 自动调整列宽
        self._auto_adjust_column_width(ws)
        
        # 保存文件
        wb.save(output_path)
    
    def _write_legend(self, ws):
        """写入颜色图例说明"""
        row = 1
        
        ws.cell(row=row, column=1, value="颜色图例 / Color Legend:")
        ws.cell(row=row, column=1).font = Font(bold=True, size=12)
        row += 1
        
        # 绿色：预测一致
        cell = ws.cell(row=row, column=1, value="绿色 (Green)")
        cell.fill = self.COLOR_SAME
        ws.cell(row=row, column=2, value="两个模型预测一致")
        row += 1
        
        # 黄色：预测不同
        cell = ws.cell(row=row, column=1, value="黄色 (Yellow)")
        cell.fill = self.COLOR_DIFF
        ws.cell(row=row, column=2, value="两个模型预测不同")
        row += 1
        
        # 浅蓝色：模型名称列
        cell = ws.cell(row=row, column=1, value="浅蓝色 (Light Blue)")
        cell.fill = self.COLOR_MODEL_NAME
        ws.cell(row=row, column=2, value="模型名称列")
        row += 1
        
        ws.cell(row=row, column=1, value="")
        row += 1
        
        ws.cell(row=row, column=1, value="说明：每个地址占两行，分别显示两个模型的预测结果，地址之间用空行分隔。")
        ws.cell(row=row, column=1).font = Font(italic=True)
    
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
        except (TypeError, ValueError, AttributeError) as e:
            # 如果自动调整失败，静默跳过
            self.logger.warning(f"Failed to auto-adjust column width: {e}")


def compare_reports(
    report1_path: str,
    report2_path: str,
    output_path: str,
    model1_name: str = "Model-1",
    model2_name: str = "Model-2",
    logger: Optional[NERLogger] = None
) -> str:
    """对比两个模型评估报告的便捷函数
    
    Args:
        report1_path: 第一个模型的报告路径
        report2_path: 第二个模型的报告路径
        output_path: 输出文件路径
        model1_name: 第一个模型名称
        model2_name: 第二个模型名称
        logger: 日志记录器
        
    Returns:
        生成的报告文件路径
    """
    comparator = ReportComparator(logger=logger)
    return comparator.generate_comparison_report(
        report1_path=report1_path,
        report2_path=report2_path,
        output_path=output_path,
        model1_name=model1_name,
        model2_name=model2_name
    )

