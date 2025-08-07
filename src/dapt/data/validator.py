"""DAPT Data Validator

Data validation utilities for DAPT training:
- Schema validation
- Data quality checks
- Format verification
- Statistical analysis
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
import logging
import re
from collections import Counter

@dataclass
class ValidationRule:
    """Data validation rule"""
    name: str
    rule_type: str  # 'required', 'format', 'range', 'custom'
    column: str
    parameters: Dict[str, Any]
    severity: str = 'error'  # 'error', 'warning', 'info'
    description: str = ''

@dataclass
class ValidationResult:
    """Validation result for a single rule"""
    rule_name: str
    passed: bool
    severity: str
    message: str
    affected_rows: List[int] = None
    statistics: Dict[str, Any] = None

class DataValidator:
    """Validator for DAPT training data"""
    
    def __init__(self, config: Dict[str, Any], global_config: Dict[str, Any]):
        self.config = config
        self.global_config = global_config
        self.country_code = config["country"]["code"]
        
        # Validation rules
        self.validation_rules: List[ValidationRule] = []
        self._setup_default_rules()
        
        # Logging
        self.logger = self._setup_logging()
        
        self.logger.info(f"Data Validator initialized for country: {self.country_code}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for validator"""
        logger = logging.getLogger(f"dapt_validator_{self.country_code}")
        logger.setLevel(getattr(logging, self.config["logging"]["level"]))
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        log_file = self.config["logging"].get("log_file")
        if log_file:
            log_path = Path(self.global_config["log_dir"]) / log_file
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def _setup_default_rules(self) -> None:
        """Setup default validation rules"""
        data_config = self.config["data"]
        label_config = self.config["labels"]
        
        text_column = data_config["text_column"]
        label_column = data_config["label_column"]
        
        # Required columns
        self.validation_rules.extend([
            ValidationRule(
                name="text_column_required",
                rule_type="required",
                column=text_column,
                parameters={},
                severity="error",
                description=f"Text column '{text_column}' must exist and not be empty"
            ),
            ValidationRule(
                name="label_column_required",
                rule_type="required",
                column=label_column,
                parameters={},
                severity="error",
                description=f"Label column '{label_column}' must exist and not be empty"
            )
        ])
        
        # Text format validation
        self.validation_rules.extend([
            ValidationRule(
                name="text_length_min",
                rule_type="range",
                column=text_column,
                parameters={
                    "min_value": data_config.get("min_text_length", 1),
                    "metric": "length"
                },
                severity="warning",
                description="Text should meet minimum length requirements"
            ),
            ValidationRule(
                name="text_length_max",
                rule_type="range",
                column=text_column,
                parameters={
                    "max_value": data_config.get("max_text_length", 512),
                    "metric": "length"
                },
                severity="warning",
                description="Text should not exceed maximum length"
            ),
            ValidationRule(
                name="text_format",
                rule_type="format",
                column=text_column,
                parameters={
                    "format_type": "text",
                    "allow_empty": False
                },
                severity="error",
                description="Text must be valid string format"
            )
        ])
        
        # Label validation
        valid_labels = label_config["label_names"]
        self.validation_rules.extend([
            ValidationRule(
                name="label_format",
                rule_type="format",
                column=label_column,
                parameters={
                    "format_type": "labels",
                    "valid_labels": valid_labels
                },
                severity="error",
                description="Labels must be in valid format and use allowed label names"
            ),
            ValidationRule(
                name="label_distribution",
                rule_type="custom",
                column=label_column,
                parameters={
                    "check_type": "distribution",
                    "min_samples_per_label": 5
                },
                severity="warning",
                description="Each label should have sufficient samples"
            )
        ])
        
        # Country-specific rules
        if self.country_code in ['ae', 'sa', 'eg', 'ma', 'lb', 'jo']:  # Arabic countries
            self.validation_rules.append(
                ValidationRule(
                    name="arabic_text_validation",
                    rule_type="custom",
                    column=text_column,
                    parameters={
                        "check_type": "arabic_text",
                        "min_arabic_ratio": 0.3
                    },
                    severity="warning",
                    description="Text should contain sufficient Arabic content for Arabic countries"
                )
            )
    
    def add_validation_rule(self, rule: ValidationRule) -> None:
        """Add a custom validation rule"""
        self.validation_rules.append(rule)
        self.logger.info(f"Added validation rule: {rule.name}")
    
    def validate_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate a pandas DataFrame"""
        self.logger.info(f"Starting validation of DataFrame with {len(df)} rows")
        
        validation_results = {
            "overall_status": "passed",
            "total_rules": len(self.validation_rules),
            "passed_rules": 0,
            "failed_rules": 0,
            "warnings": 0,
            "errors": 0,
            "rule_results": [],
            "summary": {},
            "recommendations": []
        }
        
        # Run each validation rule
        for rule in self.validation_rules:
            try:
                result = self._execute_rule(df, rule)
                validation_results["rule_results"].append(result)
                
                if result.passed:
                    validation_results["passed_rules"] += 1
                else:
                    validation_results["failed_rules"] += 1
                    
                    if result.severity == "error":
                        validation_results["errors"] += 1
                        validation_results["overall_status"] = "failed"
                    elif result.severity == "warning":
                        validation_results["warnings"] += 1
                        if validation_results["overall_status"] == "passed":
                            validation_results["overall_status"] = "passed_with_warnings"
                
            except Exception as e:
                self.logger.error(f"Error executing rule {rule.name}: {str(e)}")
                error_result = ValidationResult(
                    rule_name=rule.name,
                    passed=False,
                    severity="error",
                    message=f"Rule execution failed: {str(e)}"
                )
                validation_results["rule_results"].append(error_result)
                validation_results["failed_rules"] += 1
                validation_results["errors"] += 1
                validation_results["overall_status"] = "failed"
        
        # Generate summary and recommendations
        validation_results["summary"] = self._generate_summary(df, validation_results)
        validation_results["recommendations"] = self._generate_recommendations(validation_results)
        
        self.logger.info(f"Validation completed. Status: {validation_results['overall_status']}")
        self.logger.info(f"Passed: {validation_results['passed_rules']}, Failed: {validation_results['failed_rules']}")
        
        return validation_results
    
    def _execute_rule(self, df: pd.DataFrame, rule: ValidationRule) -> ValidationResult:
        """Execute a single validation rule"""
        if rule.rule_type == "required":
            return self._validate_required(df, rule)
        elif rule.rule_type == "format":
            return self._validate_format(df, rule)
        elif rule.rule_type == "range":
            return self._validate_range(df, rule)
        elif rule.rule_type == "custom":
            return self._validate_custom(df, rule)
        else:
            return ValidationResult(
                rule_name=rule.name,
                passed=False,
                severity="error",
                message=f"Unknown rule type: {rule.rule_type}"
            )
    
    def _validate_required(self, df: pd.DataFrame, rule: ValidationRule) -> ValidationResult:
        """Validate required fields"""
        column = rule.column
        
        # Check if column exists
        if column not in df.columns:
            return ValidationResult(
                rule_name=rule.name,
                passed=False,
                severity=rule.severity,
                message=f"Required column '{column}' is missing"
            )
        
        # Check for null/empty values
        null_mask = df[column].isnull()
        empty_mask = (df[column] == "") if df[column].dtype == 'object' else pd.Series([False] * len(df))
        invalid_mask = null_mask | empty_mask
        
        invalid_count = invalid_mask.sum()
        
        if invalid_count > 0:
            affected_rows = df.index[invalid_mask].tolist()
            return ValidationResult(
                rule_name=rule.name,
                passed=False,
                severity=rule.severity,
                message=f"Column '{column}' has {invalid_count} null/empty values",
                affected_rows=affected_rows[:100],  # Limit to first 100
                statistics={"invalid_count": invalid_count, "total_count": len(df)}
            )
        
        return ValidationResult(
            rule_name=rule.name,
            passed=True,
            severity=rule.severity,
            message=f"Column '{column}' has no missing values"
        )
    
    def _validate_format(self, df: pd.DataFrame, rule: ValidationRule) -> ValidationResult:
        """Validate data format"""
        column = rule.column
        params = rule.parameters
        format_type = params.get("format_type")
        
        if column not in df.columns:
            return ValidationResult(
                rule_name=rule.name,
                passed=False,
                severity=rule.severity,
                message=f"Column '{column}' not found"
            )
        
        if format_type == "text":
            return self._validate_text_format(df, rule)
        elif format_type == "labels":
            return self._validate_label_format(df, rule)
        else:
            return ValidationResult(
                rule_name=rule.name,
                passed=False,
                severity="error",
                message=f"Unknown format type: {format_type}"
            )
    
    def _validate_text_format(self, df: pd.DataFrame, rule: ValidationRule) -> ValidationResult:
        """Validate text format"""
        column = rule.column
        params = rule.parameters
        allow_empty = params.get("allow_empty", False)
        
        # Check data types
        non_string_mask = ~df[column].apply(lambda x: isinstance(x, str) or pd.isna(x))
        non_string_count = non_string_mask.sum()
        
        if non_string_count > 0:
            return ValidationResult(
                rule_name=rule.name,
                passed=False,
                severity=rule.severity,
                message=f"Column '{column}' has {non_string_count} non-string values",
                affected_rows=df.index[non_string_mask].tolist()[:100]
            )
        
        # Check for empty strings if not allowed
        if not allow_empty:
            empty_mask = (df[column] == "") & df[column].notna()
            empty_count = empty_mask.sum()
            
            if empty_count > 0:
                return ValidationResult(
                    rule_name=rule.name,
                    passed=False,
                    severity=rule.severity,
                    message=f"Column '{column}' has {empty_count} empty strings",
                    affected_rows=df.index[empty_mask].tolist()[:100]
                )
        
        return ValidationResult(
            rule_name=rule.name,
            passed=True,
            severity=rule.severity,
            message=f"Column '{column}' has valid text format"
        )
    
    def _validate_label_format(self, df: pd.DataFrame, rule: ValidationRule) -> ValidationResult:
        """Validate label format"""
        column = rule.column
        params = rule.parameters
        valid_labels = params.get("valid_labels", [])
        
        invalid_rows = []
        invalid_labels = set()
        
        for idx, label_data in df[column].items():
            if pd.isna(label_data):
                invalid_rows.append(idx)
                continue
            
            # Parse labels based on format
            try:
                if isinstance(label_data, str):
                    try:
                        # Try JSON parsing
                        labels = json.loads(label_data)
                        if not isinstance(labels, list):
                            labels = [labels]
                    except:
                        # Single string label
                        labels = [label_data]
                elif isinstance(label_data, list):
                    labels = label_data
                else:
                    # Numeric or other format
                    labels = [label_data]
                
                # Check if all labels are valid
                for label in labels:
                    if label not in valid_labels:
                        invalid_labels.add(label)
                        if idx not in invalid_rows:
                            invalid_rows.append(idx)
                            
            except Exception:
                invalid_rows.append(idx)
        
        if invalid_rows:
            message = f"Column '{column}' has {len(invalid_rows)} rows with invalid labels"
            if invalid_labels:
                message += f". Invalid labels found: {list(invalid_labels)[:10]}"
            
            return ValidationResult(
                rule_name=rule.name,
                passed=False,
                severity=rule.severity,
                message=message,
                affected_rows=invalid_rows[:100],
                statistics={"invalid_labels": list(invalid_labels)}
            )
        
        return ValidationResult(
            rule_name=rule.name,
            passed=True,
            severity=rule.severity,
            message=f"Column '{column}' has valid label format"
        )
    
    def _validate_range(self, df: pd.DataFrame, rule: ValidationRule) -> ValidationResult:
        """Validate value ranges"""
        column = rule.column
        params = rule.parameters
        metric = params.get("metric", "value")
        min_value = params.get("min_value")
        max_value = params.get("max_value")
        
        if column not in df.columns:
            return ValidationResult(
                rule_name=rule.name,
                passed=False,
                severity=rule.severity,
                message=f"Column '{column}' not found"
            )
        
        # Calculate metric values
        if metric == "length":
            values = df[column].astype(str).str.len()
        elif metric == "value":
            values = pd.to_numeric(df[column], errors='coerce')
        else:
            return ValidationResult(
                rule_name=rule.name,
                passed=False,
                severity="error",
                message=f"Unknown metric: {metric}"
            )
        
        # Check ranges
        invalid_mask = pd.Series([False] * len(df))
        
        if min_value is not None:
            invalid_mask |= (values < min_value)
        
        if max_value is not None:
            invalid_mask |= (values > max_value)
        
        invalid_count = invalid_mask.sum()
        
        if invalid_count > 0:
            affected_rows = df.index[invalid_mask].tolist()
            range_str = f"[{min_value if min_value is not None else '-∞'}, {max_value if max_value is not None else '∞'}]"
            
            return ValidationResult(
                rule_name=rule.name,
                passed=False,
                severity=rule.severity,
                message=f"Column '{column}' has {invalid_count} values outside range {range_str}",
                affected_rows=affected_rows[:100],
                statistics={
                    "invalid_count": invalid_count,
                    "min_found": float(values.min()),
                    "max_found": float(values.max()),
                    "expected_range": range_str
                }
            )
        
        return ValidationResult(
            rule_name=rule.name,
            passed=True,
            severity=rule.severity,
            message=f"Column '{column}' values are within expected range"
        )
    
    def _validate_custom(self, df: pd.DataFrame, rule: ValidationRule) -> ValidationResult:
        """Validate custom rules"""
        column = rule.column
        params = rule.parameters
        check_type = params.get("check_type")
        
        if check_type == "distribution":
            return self._validate_label_distribution(df, rule)
        elif check_type == "arabic_text":
            return self._validate_arabic_text(df, rule)
        else:
            return ValidationResult(
                rule_name=rule.name,
                passed=False,
                severity="error",
                message=f"Unknown custom check type: {check_type}"
            )
    
    def _validate_label_distribution(self, df: pd.DataFrame, rule: ValidationRule) -> ValidationResult:
        """Validate label distribution"""
        column = rule.column
        params = rule.parameters
        min_samples = params.get("min_samples_per_label", 5)
        
        # Collect all labels
        all_labels = []
        for label_data in df[column].dropna():
            try:
                if isinstance(label_data, str):
                    try:
                        labels = json.loads(label_data)
                        if not isinstance(labels, list):
                            labels = [labels]
                    except:
                        labels = [label_data]
                elif isinstance(label_data, list):
                    labels = label_data
                else:
                    labels = [label_data]
                
                all_labels.extend(labels)
            except:
                continue
        
        # Count label distribution
        label_counts = Counter(all_labels)
        
        # Find labels with insufficient samples
        insufficient_labels = [label for label, count in label_counts.items() if count < min_samples]
        
        if insufficient_labels:
            return ValidationResult(
                rule_name=rule.name,
                passed=False,
                severity=rule.severity,
                message=f"Labels with insufficient samples (< {min_samples}): {insufficient_labels}",
                statistics={
                    "label_distribution": dict(label_counts),
                    "insufficient_labels": insufficient_labels,
                    "min_required": min_samples
                }
            )
        
        return ValidationResult(
            rule_name=rule.name,
            passed=True,
            severity=rule.severity,
            message="Label distribution is adequate",
            statistics={"label_distribution": dict(label_counts)}
        )
    
    def _validate_arabic_text(self, df: pd.DataFrame, rule: ValidationRule) -> ValidationResult:
        """Validate Arabic text content"""
        column = rule.column
        params = rule.parameters
        min_arabic_ratio = params.get("min_arabic_ratio", 0.3)
        
        arabic_pattern = re.compile(r'[\u0600-\u06FF]')
        
        low_arabic_rows = []
        
        for idx, text in df[column].items():
            if pd.isna(text) or not isinstance(text, str):
                continue
            
            # Count Arabic characters
            arabic_chars = len(arabic_pattern.findall(text))
            total_chars = len(text.replace(' ', ''))  # Exclude spaces
            
            if total_chars > 0:
                arabic_ratio = arabic_chars / total_chars
                if arabic_ratio < min_arabic_ratio:
                    low_arabic_rows.append(idx)
        
        if low_arabic_rows:
            return ValidationResult(
                rule_name=rule.name,
                passed=False,
                severity=rule.severity,
                message=f"{len(low_arabic_rows)} texts have low Arabic content (< {min_arabic_ratio:.1%})",
                affected_rows=low_arabic_rows[:100],
                statistics={
                    "low_arabic_count": len(low_arabic_rows),
                    "min_required_ratio": min_arabic_ratio
                }
            )
        
        return ValidationResult(
            rule_name=rule.name,
            passed=True,
            severity=rule.severity,
            message="Arabic text content is adequate"
        )
    
    def _generate_summary(self, df: pd.DataFrame, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate validation summary"""
        summary = {
            "dataset_size": len(df),
            "columns": list(df.columns),
            "validation_status": validation_results["overall_status"],
            "error_count": validation_results["errors"],
            "warning_count": validation_results["warnings"],
            "data_quality_score": 0.0
        }
        
        # Calculate data quality score (0-100)
        total_rules = validation_results["total_rules"]
        passed_rules = validation_results["passed_rules"]
        
        if total_rules > 0:
            base_score = (passed_rules / total_rules) * 100
            
            # Penalize errors more than warnings
            error_penalty = validation_results["errors"] * 10
            warning_penalty = validation_results["warnings"] * 5
            
            summary["data_quality_score"] = max(0, base_score - error_penalty - warning_penalty)
        
        return summary
    
    def _generate_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        # Analyze failed rules
        for result in validation_results["rule_results"]:
            if not result.passed:
                if "missing" in result.message.lower():
                    recommendations.append(f"Add missing data for: {result.message}")
                elif "invalid" in result.message.lower():
                    recommendations.append(f"Fix invalid data: {result.message}")
                elif "range" in result.message.lower():
                    recommendations.append(f"Adjust data to meet range requirements: {result.message}")
                elif "distribution" in result.message.lower():
                    recommendations.append(f"Balance dataset: {result.message}")
        
        # General recommendations based on quality score
        quality_score = validation_results.get("summary", {}).get("data_quality_score", 0)
        
        if quality_score < 50:
            recommendations.append("Data quality is poor. Consider comprehensive data cleaning.")
        elif quality_score < 80:
            recommendations.append("Data quality is moderate. Address major issues before training.")
        
        # Remove duplicates
        recommendations = list(set(recommendations))
        
        return recommendations
    
    def save_validation_report(self, validation_results: Dict[str, Any], output_path: str) -> bool:
        """Save validation report to file"""
        try:
            report_path = Path(output_path)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert ValidationResult objects to dictionaries
            serializable_results = validation_results.copy()
            serializable_results["rule_results"] = [
                {
                    "rule_name": result.rule_name,
                    "passed": result.passed,
                    "severity": result.severity,
                    "message": result.message,
                    "affected_rows_count": len(result.affected_rows) if result.affected_rows else 0,
                    "statistics": result.statistics
                }
                for result in validation_results["rule_results"]
            ]
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Validation report saved to: {report_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving validation report: {str(e)}")
            return False