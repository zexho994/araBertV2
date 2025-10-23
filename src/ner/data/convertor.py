"""CSV Annotation Generator

Reads a CSV file with a formatted address column and entity columns (lowercased
as defined in the country config), then generates BIO-tagged annotations per row
and saves them under the country-specific data directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, NamedTuple

import csv
import json
import re

import pandas as pd

from ..config.manager import ConfigManager

class ValidationIssue(NamedTuple):
    """表示单个校验问题"""
    row_index: int
    entity_column: str
    entity_value: str
    text_value: str
    issue_type: str


class ValidationReport(NamedTuple):
    """校验报告"""
    total_rows: int
    valid_rows: int
    invalid_rows: int
    issues: List[ValidationIssue]
    success_rate: float


@dataclass
class CSVConvertConfig:
    """
    转换配置
    Args:
        csv_path: CSV文件路径
        country_code: 国家代码
        output_file: 输出文件路径
        text_column: 文本列名
        validate_text_contains_entities: 是否校验text_column包含实体列的文本
        validation_mode: 校验模式
        auto_process: 是否处理异常数据（删除异常数据行）
    """
    csv_path: str
    country_code: str
    output_file: Optional[str] = None  # If None, defaults to data dir / country / generated.json
    text_column: str = "formatted_address"
    validate_text_contains_entities: bool = True  # 是否校验text_column包含实体列的文本
    validation_mode: str = "strict"  # "strict" 或 "lenient"，严格模式会阻止生成，宽松模式只警告
    auto_process: bool = False

class CSVAnnotationConvert:
    """Generate NER annotations from a CSV using country-specific labels.

    CSV requirements:
    - Must contain a text column (default: "formatted_address")
    - Must contain one or more entity columns whose names are the lowercase
      forms of the entity labels configured for the country (without BIO prefixes)
      e.g., for entity CITY -> column name "city".

    Output format:
    - Defaults to JSON Lines (JSONL), one example per line, saved under
      data/ner/data/<country_code>/generated.jsonl
    - If output_file ends with .json, a JSON array will be written instead.
    """

    def __init__(self, config_dir: str = "data/ner/configs") -> None:
        self.config_manager = ConfigManager(config_dir=config_dir)
        self._abbrev_synonyms: Dict[str, List[str]] = {
            "street": ["st", "st."],
            "st": ["st", "st."],

            "road": ["rd", "rd."],
            "rd": ["road", "rd."],

            "north": ["n"],
            "west": ["w"],
            "south": ["s"],
            "east": ["e"],
            "suburb": ["area"],
            "Alhamriya": ["al hamriya"],

            "avenue": ["avenue", "ave", "ave."],
            "boulevard": ["boulevard", "blvd", "blvd."],
            "drive": ["drive", "dr", "dr."],
            "lane": ["lane", "ln", "ln."],
            "court": ["court", "ct", "ct."],
            "place": ["place", "pl", "pl."],
            "square": ["square", "sq", "sq."],
            "terrace": ["terrace", "ter", "ter."],
            "highway": ["highway", "hwy", "hwy."],
            "parkway": ["parkway", "pkwy", "pkwy."],
            "building": ["building", "bldg", "bldg."],
            "united arab emirates": ["uae"],
            "mount": ["mt"],
            "sheikh": ["sheikh","shk","shk."],
            "al-khaimah": ["al-khaimah","al khaimah"],
            "ajman": ["ajman"] #ajman en , arabic
        }

        self._entity_priority: Dict[str, int] = {
            "BUILDING": 100,
            "STREET": 90,
            "COMPOUND": 80,
            "SUB_AREA": 70,
            "EMIRATE": 60,
            "CITY": 50,
            "COUNTRY": 40,
        }

    def generate(self, cfg: CSVConvertConfig) -> Path:
        """生成jsonl训练数据集
        
        Args:
            cfg: 转换配置
        
        Returns:
            Path: 输出文件路径
        """
        country_cfg = self.config_manager.load_country_config(cfg.country_code)
        entity_names = self._derive_entity_names_from_config(country_cfg)
        entity_to_column = {e: e.lower() for e in entity_names}

        # Read CSV with delimiter auto-detection and clean headers
        df = pd.read_csv(cfg.csv_path, sep=None, engine="python")
        # Remove BOM and surrounding whitespace from headers
        df.columns = [str(c).lstrip("\ufeff").strip() for c in df.columns]
        # Build normalized header map
        norm_to_actual = {self._normalize_colname(c): c for c in df.columns}

        norm_text_col = self._normalize_colname(cfg.text_column)
        if norm_text_col not in norm_to_actual:
            raise ValueError(
                f"CSV missing required text column '{cfg.text_column}'. Columns: {list(df.columns)}"
            )
        text_col_actual = norm_to_actual[norm_text_col]

        # 将期望的实体列名映射到实际的列名
        entity_to_actual_col: Dict[str, str] = {}
        for entity, expected_col in entity_to_column.items():
            norm_expected = self._normalize_colname(expected_col)
            if norm_expected in norm_to_actual:
                entity_to_actual_col[entity] = norm_to_actual[norm_expected]

        present_entity_columns = list(entity_to_actual_col.values())
        if not present_entity_columns:
            raise ValueError(
                "CSV does not contain any expected entity columns. "
                f"Expected any of: {sorted(entity_to_column.values())}. Columns: {list(df.columns)}"
            )

        # 执行CSV校验（如果启用）
        validation_report = None
        if cfg.validate_text_contains_entities:
            print("正在执行CSV校验...")
            validation_report = self.validate_csv_text_contains_entities(
                df, entity_to_actual_col, text_col_actual
            )
            # 生成校验报告CSV文件
            csv_path = Path(cfg.csv_path)
            report_csv_path = csv_path.parent / f"{csv_path.stem}_validation_report.csv"
            self.print_validation_report(validation_report, str(report_csv_path), show_details=True)
            
            # 根据校验模式决定是否继续
            if cfg.validation_mode == "strict" and validation_report.success_rate < 1.0:
                raise ValueError(
                    f"CSV校验失败: 成功率 {validation_report.success_rate:.2%} < 100%。"
                    f"发现 {len(validation_report.issues)} 个问题。"
                    "请修复数据问题或使用 validation_mode='lenient' 继续处理。"
                )
            elif cfg.validation_mode == "lenient" and validation_report.issues:
                print(f"\n⚠️  警告: 发现 {len(validation_report.issues)} 个数据质量问题，但在宽松模式下继续处理。")

        dynamic_priority: Dict[str, int] = {}
        for entity, actual_col in entity_to_actual_col.items():
            try:
                col_index = list(df.columns).index(actual_col)
                # Base priority starts from 100, with column index as priority
                # Later columns (higher index) get higher priority
                dynamic_priority[entity.upper()] = 100 + col_index
            except ValueError:
                # Fallback to default priority if column not found
                dynamic_priority[entity.upper()] = self._entity_priority.get(entity.upper(), 0)

        examples: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            text = str(row.get(text_col_actual, "")).strip()
            if not text:
                continue

            tokens, token_spans = self._whitespace_tokenize_with_spans(text)
            entities = self._extract_entities_from_row(
                row=row,
                entity_to_column=entity_to_actual_col,
                tokens=tokens,
                token_spans=token_spans,
                original_text=text,
            )

            labels = self._convert_entities_to_bio_with_priority(tokens, entities, dynamic_priority)
            examples.append({
                "text": text,
                "tokens": tokens,
                "labels": labels,
            })

        # Determine output path (default to JSONL)
        default_dir = Path("data/ner/data") / cfg.country_code
        output_path = Path(cfg.output_file) if cfg.output_file else default_dir / "generated.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write according to extension
        suffix = output_path.suffix.lower()
        if suffix == ".jsonl":
            with output_path.open("w", encoding="utf-8") as f:
                for ex in examples:
                    f.write(json.dumps(ex, ensure_ascii=False))
                    f.write("\n")
        else:
            with (output_path.with_suffix(".jsonl")).open("w", encoding="utf-8") as f:
                for ex in examples:
                    f.write(json.dumps(ex, ensure_ascii=False))
                    f.write("\n")

        return output_path

    def validate_csv_text_contains_entities(
        self, 
        df: pd.DataFrame, 
        entity_to_actual_col: Dict[str, str],
        text_col_actual: str
    ) -> ValidationReport:
        """校验CSV中的text_column是否包含其他实体列中的文本
        
        Args:
            cfg: 配置对象
            df: CSV数据框
            entity_to_actual_col: 实体名到实际列名的映射
            text_col_actual: 实际的文本列名
            
        Returns:
            ValidationReport: 校验报告
        """
        issues = []
        valid_rows = 0
        
        for row_idx, row in df.iterrows():
            text = str(row.get(text_col_actual, "")).strip()
            if not text:
                continue
                
            row_has_issues = False
            
            # 检查每个实体列
            for _, actual_col in entity_to_actual_col.items():
                if actual_col not in row or pd.isna(row[actual_col]):
                    continue
                    
                entity_value = str(row[actual_col]).strip()
                if not entity_value:
                    continue
                
                # 支持多个值用 | 分隔
                parts = [p.strip() for p in re.split(r"\s*\|\s*", entity_value) if p.strip()]
                
                for part in parts:
                    # 使用现有的模糊匹配方法来查找实体
                    char_spans = self._find_all_char_spans(text, part)
                    
                    if not char_spans:
                        # 未找到匹配
                        issues.append(ValidationIssue(
                            row_index=int(row_idx),
                            entity_column=actual_col,
                            entity_value=part,
                            text_value=text,
                            issue_type="not_found"
                        ))
                        row_has_issues = True
            
            if not row_has_issues:
                valid_rows += 1
        
        total_rows = len(df)
        invalid_rows = total_rows - valid_rows
        success_rate = valid_rows / total_rows if total_rows > 0 else 0.0
        
        return ValidationReport(
            total_rows=total_rows,
            valid_rows=valid_rows,
            invalid_rows=invalid_rows,
            issues=issues,
            success_rate=success_rate
        )

    def process_validation_data(self, csv_path: str, df: pd.DataFrame, validation_report: ValidationReport, 
                               cfg: CSVConvertConfig, entity_to_actual_col: Dict[str, str], 
                               text_col_actual: str) -> ValidationReport:
        """处理异常数据
        
        Args:
            csv_path: 原CSV文件路径
            df: 原始数据框
            validation_report: 校验报告
            cfg: 配置对象
            entity_to_actual_col: 实体到列名的映射
            text_col_actual: 实际文本列名
            
        Returns:
            ValidationReport: 处理后的校验报告
        """
        print("开始处理异常数据...")
        
        # 获取所有NOT_FOUND问题的行索引
        not_found_rows = set()
        for issue in validation_report.issues:
            if issue.issue_type == "not_found":
                not_found_rows.add(issue.row_index)
        
        if not_found_rows:
            print(f"发现 {len(not_found_rows)} 行包含NOT_FOUND问题，即将删除这些行")
            
            # 删除问题行
            df_cleaned = df.drop(index=list(not_found_rows)).reset_index(drop=True)
            
            # 生成处理后的CSV文件
            csv_path_obj = Path(csv_path)
            processed_csv_path = csv_path_obj.parent / f"{csv_path_obj.stem}_processed{csv_path_obj.suffix}"
            df_cleaned.to_csv(processed_csv_path, index=False, encoding='utf-8')
            print(f"处理后的CSV文件已保存: {processed_csv_path}")
            print(f"原始行数: {len(df)}, 处理后行数: {len(df_cleaned)}, 删除行数: {len(not_found_rows)}")
            
            # 重新执行校验
            CSVConvertConfig(
                csv_path=str(processed_csv_path),
                country_code=cfg.country_code,
                text_column=cfg.text_column,
            )
            
            new_validation_report = self.validate_csv_text_contains_entities(
                df_cleaned, entity_to_actual_col, text_col_actual
            )
            
            print(f"处理后校验结果 - 总行数: {new_validation_report.total_rows}, "
                  f"有效行数: {new_validation_report.valid_rows}, "
                  f"成功率: {new_validation_report.success_rate:.2%}")
            
            return new_validation_report
        else:
            print("未发现NOT_FOUND问题，无需处理")
            return validation_report

    def print_validation_report(self, report: ValidationReport, csv_file_path: str, show_details: bool = True):
        """生成CSV格式的校验报告
        
        Args:
            report: 校验报告对象
            csv_file_path: CSV文件输出路径
            show_details: 是否包含详细问题信息
        """
        # 确保输出目录存在
        output_path = Path(csv_file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 准备CSV数据
        csv_data = []
        
        # 添加汇总信息作为CSV的前几行
        csv_data.append(['报告类型', '数值', '说明'])
        csv_data.append(['总行数', report.total_rows, '输入CSV文件的总行数'])
        csv_data.append(['有效行数', report.valid_rows, '通过校验的行数'])
        csv_data.append(['问题行数', report.invalid_rows, '存在问题的行数'])
        csv_data.append(['成功率', f"{report.success_rate:.2%}", '有效行数/总行数'])
        csv_data.append(['问题总数', len(report.issues), '发现的问题总数'])
        csv_data.append(['', '', ''])  # 空行分隔
        
        if show_details and report.issues:
            # 添加详细问题列表
            csv_data.append(['问题序号', '行索引', '实体列名', '实体值', '问题类型', '文本内容'])
            
            # 按问题类型分组并排序
            issues_by_type = {}
            for issue in report.issues:
                if issue.issue_type not in issues_by_type:
                    issues_by_type[issue.issue_type] = []
                issues_by_type[issue.issue_type].append(issue)
            
            problem_count = 1
            for issue_type in sorted(issues_by_type.keys()):
                type_issues = issues_by_type[issue_type]
                
                # 添加问题类型分隔行
                csv_data.append([f"=== {issue_type.upper()} ({len(type_issues)} 个问题) ===", '', '', '', '', ''])
                
                for issue in type_issues:
                    # 截断过长的文本以便在CSV中查看
                    text_content = issue.text_value
                    if len(text_content) > 200:
                        text_content = text_content[:200] + "..."
                    
                    csv_data.append([
                        problem_count,
                        issue.row_index+2,
                        issue.entity_column,
                        issue.entity_value,
                        issue.issue_type,
                        text_content
                    ])
                    problem_count += 1
                
                # 问题类型之间添加空行
                csv_data.append(['', '', '', '', '', ''])
        
        # 写入CSV文件
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(csv_data)
        
        print(f"校验报告已生成: {csv_file_path}")
        print(f"总行数: {report.total_rows}, 有效行数: {report.valid_rows}, 成功率: {report.success_rate:.2%}")
        
        if report.issues:
            print(f"发现 {len(report.issues)} 个问题，详细信息请查看CSV文件")

    def generate_from_csv(self, csv_path: str, country_code: str, *, output_file: Optional[str] = None,
                          text_column: str = "formatted_address", 
                          validate_text_contains_entities: bool = True,
                          validation_mode: str = "strict") -> Path:
        cfg = CSVConvertConfig(
            csv_path=csv_path,
            country_code=country_code,
            output_file=output_file,
            text_column=text_column,
            validate_text_contains_entities=validate_text_contains_entities,
            validation_mode=validation_mode,
        )
        return self.generate(cfg)

    def validate_csv(self, csv_path: str, country_code: str, *, 
                         text_column: str = "formatted_address", 
                         auto_process: bool = False) -> ValidationReport:
        """执行CSV校验，可选择自动处理异常数据
        
        Args:
            csv_path: CSV文件路径
            country_code: 国家代码
            text_column: 文本列名
            auto_process: 是否自动处理异常数据（删除NOT_FOUND数据行）
            
        Returns:
            ValidationReport: 校验报告
        """
        # Load country config and derive entity set
        country_cfg = self.config_manager.load_country_config(country_code)
        entity_names = self._derive_entity_names_from_config(country_cfg)
        entity_to_column = {e: e.lower() for e in entity_names}

        # Read CSV with delimiter auto-detection and clean headers
        df = pd.read_csv(csv_path, sep=None, engine="python")
        df.columns = [str(c).lstrip("\ufeff").strip() for c in df.columns]
        norm_to_actual = {self._normalize_colname(c): c for c in df.columns}

        # Validate columns
        norm_text_col = self._normalize_colname(text_column)
        if norm_text_col not in norm_to_actual:
            raise ValueError(
                f"CSV missing required text column '{text_column}'. Columns: {list(df.columns)}"
            )
        text_col_actual = norm_to_actual[norm_text_col]

        # Map expected entity lower names to actual present column names
        entity_to_actual_col: Dict[str, str] = {}
        for entity, expected_col in entity_to_column.items():
            norm_expected = self._normalize_colname(expected_col)
            if norm_expected in norm_to_actual:
                entity_to_actual_col[entity] = norm_to_actual[norm_expected]

        if not entity_to_actual_col:
            raise ValueError(
                "CSV does not contain any expected entity columns. "
                f"Expected any of: {sorted(entity_to_column.values())}. Columns: {list(df.columns)}"
            )

        # Create a dummy config for validation
        cfg = CSVConvertConfig(
            csv_path=csv_path,
            country_code=country_code,
            text_column=text_column,
        )

        # Execute validation
        validation_report = self.validate_csv_text_contains_entities(df, entity_to_actual_col, text_col_actual)
        
        # Auto-process data if requested
        if auto_process and validation_report.issues:
            processed_report = self.process_validation_data(
                csv_path, df, validation_report, cfg, entity_to_actual_col, text_col_actual
            )
            return processed_report
        
        return validation_report

    @staticmethod
    def _derive_entity_names_from_config(country_cfg: Dict[str, Any]) -> List[str]:
        """Derive entity names from config labels section.

        Supports two shapes:
        - labels.entities = ["CITY", "STREET", ...]
        - labels.label_names = ["O", "B-CITY", "I-CITY", ...]
        """
        labels_cfg = country_cfg.get("labels", {})

        entities: List[str] = []
        if isinstance(labels_cfg.get("entities"), list) and labels_cfg["entities"]:
            entities = [str(e).upper() for e in labels_cfg["entities"]]
        elif isinstance(labels_cfg.get("label_names"), list) and labels_cfg["label_names"]:
            uniq: Dict[str, None] = {}
            for name in labels_cfg["label_names"]:
                if not isinstance(name, str):
                    continue
                if name == "O":
                    continue
                if name.startswith("B-") or name.startswith("I-"):
                    ent = name[2:].upper()
                else:
                    ent = name.upper()
                uniq[ent] = None
            entities = list(uniq.keys())
        else:
            raise ValueError("Could not derive entity names from config.labels")

        return entities

    @staticmethod
    def _whitespace_tokenize_with_spans(text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
        """Tokenize on whitespace and return tokens and (start,end) char spans for each token."""
        tokens: List[str] = []
        spans: List[Tuple[int, int]] = []
        for m in re.finditer(r"\S+", text):
            token = m.group(0)
            tokens.append(token)
            spans.append((m.start(), m.end()))
        return tokens, spans

    def _find_char_span(self, text: str, value: str) -> Optional[Tuple[int, int]]:
        """Return the first (start,end) char span of value in text using fuzzy matching.

        Strategy:
        - Build a case-insensitive regex from the provided value where known tokens
          (e.g., "Street") expand to alternatives (e.g., "Street|St|St.").
        - Allow flexible separators between tokens (spaces/hyphens).
        - Fallback to exact substring search if regex fails.
        """
        if not isinstance(value, str):
            return None
        s = value.strip()
        if not s:
            return None

        pattern = self._build_value_regex(s)
        m = pattern.search(text)
        if m:
            return m.start(), m.end()

        # Fallback to exact find (case-insensitive)
        idx = text.lower().find(s.lower())
        if idx == -1:
            return None
        return idx, idx + len(s)
    
    def _find_all_char_spans(self, text: str, value: str) -> List[Tuple[int, int]]:
        """Return all (start,end) char spans of value in text using fuzzy matching.

        Strategy:
        - Build a case-insensitive regex from the provided value where known tokens
          (e.g., "Street") expand to alternatives (e.g., "Street|St|St.").
        - Allow flexible separators between tokens (spaces/hyphens).
        - Fallback to exact substring search if regex fails.
        """
        if not isinstance(value, str):
            return []
        s = value.strip()
        if not s:
            return []

        spans = []
        pattern = self._build_value_regex(s)
        
        # Find all regex matches
        for m in pattern.finditer(text):
            spans.append((m.start(), m.end()))
        
        # If no regex matches, fallback to exact substring search
        if not spans:
            text_lower = text.lower()
            s_lower = s.lower()
            start = 0
            while True:
                idx = text_lower.find(s_lower, start)
                if idx == -1:
                    break
                spans.append((idx, idx + len(s)))
                start = idx + 1
        
        return spans

    @staticmethod
    def _normalize_colname(name: str) -> str:
        """
        规范化列名
        Args:
            name: 列名
        Returns:
            str: 规范化后的列名
        """
        if name is None:
            return ""
        s = str(name).lstrip("\ufeff").strip().lower()
        # 替换空白和连字符为下划线
        s = re.sub(r"[\s\-]+", "_", s)
        # 合并多个下划线
        s = re.sub(r"_+", "_", s)
        return s

    def _extract_entities_from_row(
        self,
        row: pd.Series,
        entity_to_column: Dict[str, str],
        tokens: List[str],
        token_spans: List[Tuple[int, int]],
        original_text: str,
    ) -> List[Dict[str, Any]]:
        """Extract entities as token-span dicts: {type,start,end} (end exclusive)."""
        text_values: Dict[str, str] = {}
        for entity, col in entity_to_column.items():
            if col in row and pd.notna(row[col]):
                val = str(row[col]).strip()
                if val:
                    text_values[entity] = val

        entities: List[Dict[str, Any]] = []

        for entity, value in text_values.items():
            # Support multiple values separated by |
            parts = [p.strip() for p in re.split(r"\s*\|\s*", value) if p.strip()]
            for part in parts:
                # Find all occurrences of this part in the text
                char_spans = self._find_all_char_spans(original_text, part)
                for start_c, end_c in char_spans:
                    # Map char span to token indices
                    start_tok = None
                    end_tok = None
                    for i, (s, e) in enumerate(token_spans):
                        if start_tok is None and s <= start_c < e:
                            start_tok = i
                        if start_tok is not None and s < end_c <= e:
                            end_tok = i + 1  # exclusive
                            break

                    if start_tok is None:
                        # If entity starts between tokens (e.g., punctuation), approximate to nearest token
                        for i, (s, e) in enumerate(token_spans):
                            if s >= start_c:
                                start_tok = i
                                break
                        if start_tok is None:
                            start_tok = 0

                    if end_tok is None:
                        for i, (s, e) in enumerate(token_spans[start_tok:], start_tok):
                            if e >= end_c:
                                end_tok = i + 1
                                break
                        if end_tok is None:
                            end_tok = len(tokens)

                    if 0 <= start_tok < end_tok <= len(tokens):
                        entities.append({
                            "type": entity,
                            "start": start_tok,
                            "end": end_tok,
                        })

        return entities

    # --- Helper methods for fuzzy matching and BIO conversion with priority ---

    def _build_value_regex(self, value: str) -> re.Pattern:
        """Build a fuzzy, case-insensitive regex that matches the given value in text.

        - Escapes literal tokens except where a token has known abbreviations/synonyms
          which are expanded into a non-capturing group.
        - Joins tokens with a flexible separator pattern to tolerate spaces/hyphens.
        """
        # Split by whitespace to tokens
        raw_tokens = [t for t in re.split(r"\s+", value.strip()) if t]
        regex_tokens: List[str] = []
        for tok in raw_tokens:
            base = tok.strip().strip(',')
            key = base.lower().rstrip('.')
            if key in self._abbrev_synonyms:
                alts = self._abbrev_synonyms[key]
                # Escape each alt for regex; allow optional trailing dot variants
                escaped_alts = [re.escape(a) for a in alts]
                regex_tokens.append(f"(?:{'|'.join(escaped_alts)})")
            else:
                regex_tokens.append(re.escape(base))

        # Allow spaces and hyphens between tokens
        sep = r"[\s\-]+"
        pattern_str = sep.join(regex_tokens)
        # Use word boundary on ends to avoid partial matches when possible
        pattern_str = rf"\b{pattern_str}\b"
        try:
            return re.compile(pattern_str, flags=re.IGNORECASE)
        except re.error:
            # Extremely defensive: fall back to escaped literal
            return re.compile(re.escape(value), flags=re.IGNORECASE)

    def _convert_entities_to_bio_with_priority(
        self,
        tokens: List[str],
        entities: List[Dict[str, Any]],
        dynamic_priority: Optional[Dict[str, int]] = None,
    ) -> List[str]:
        """Convert entities to BIO labels without overlaps, honoring entity priority.

        - Sort entities by (priority desc, span_length desc) to place higher priority
          and longer entities first.
        - Only place an entity if its entire span is currently unlabeled (all 'O').
        - Uses dynamic_priority if provided, otherwise falls back to self._entity_priority.
        """
        labels: List[str] = ["O"] * len(tokens)

        def priority_for(entity_type: str) -> int:
            if dynamic_priority:
                return dynamic_priority.get(entity_type.upper(), 0)
            return self._entity_priority.get(entity_type.upper(), 0)

        def span_len(ent: Dict[str, Any]) -> int:
            return int(ent.get("end", 0)) - int(ent.get("start", 0))

        sorted_entities = sorted(
            entities,
            key=lambda e: (priority_for(e.get("type", "")), span_len(e)),
            reverse=True,
        )

        for ent in sorted_entities:
            start = int(ent.get("start", 0))
            end = int(ent.get("end", 0))
            etype = str(ent.get("type", "")).upper()
            if not (0 <= start < end <= len(tokens)):
                continue

            # Check if span is free (all 'O')
            if any(label != "O" for label in labels[start:end]):
                continue

            labels[start] = f"B-{etype}"
            for i in range(start + 1, end):
                labels[i] = f"I-{etype}"

        return labels


def main():
    gen = CSVAnnotationConvert(config_dir="data/ner/configs")
    
    # csv_path = "src/ner/utils/uae_train_0910.csv"
    # out_path = "data/ner/data/uae/train_250910.jsonl"

    # csv_path = "src/ner/utils/uae_eval_0910.csv"
    # out_path = "data/ner/data/uae/eval_250910.jsonl"

    # 示例1: 仅执行校验
    # print("=== 示例1: 仅执行CSV校验 ===")
    # validation_report = gen.validate_csv(
    #     csv_path=csv_path,
    #     country_code="uae",
    #     text_column="formatted_address",
    #     auto_process=True #是否删除异常数据行
    # )
    # csv_input_path = Path(csv_path)
    # report_csv_path = csv_input_path.parent / f"{csv_input_path.stem}_validation_report.csv"
    # gen.print_validation_report(validation_report, str(report_csv_path), show_details=True)

    # 示例2: 生成训练文件（带校验）
    # print("\n=== 示例2: 生成注释文件（严格模式校验） ===")
    # try:
    #     output = gen.generate_from_csv(
    #         csv_path=csv_path,
    #         country_code="uae",
    #         output_file=out_path,
    #         validate_text_contains_entities=True,
    #         validation_mode="strict"  # 严格模式：如果校验失败会抛出异常
    #     )
    #     print(f"生成成功: {output}")
    # except ValueError as e:
    #     print(f"严格模式校验失败: {e}")
        

if __name__ == "__main__":
    main()