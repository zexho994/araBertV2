"""数据处理相关命令

支持：
- 数据校验（validate）
- 数据预处理（process）
- 数据转换（convert）- CSV转JSONL格式
- 数据划分（split，暂未实现）

# TODO: 实现 `split`，并支持自定义随机种子与分层抽样。
"""

from pathlib import Path

from .base import BaseCommand

class DataCommand(BaseCommand):
    """数据处理相关命令

    支持：
    - 数据校验（validate）
    - 数据预处理（process）
    - 数据转换（convert）- CSV转JSONL格式
    - 数据划分（split，暂未实现）

    # TODO: 实现 `split`，并支持自定义随机种子与分层抽样。
    """
    
    @property
    def name(self) -> str:
        return "data"
    
    @property
    def description(self) -> str:
        return "Process and validate training data"
    
    def setup_parser(self, parser):
        subparsers = parser.add_subparsers(dest="data_action", help="Data actions")
        
        # 验证训练数据
        validate_parser = subparsers.add_parser("validate", help="Validate training/evaluation data")
        validate_parser.add_argument("--country", required=True, help="Country code")
        validate_parser.add_argument("--input-file", required=True, help="Data file to validate")
        
        # 处理训练数据
        process_parser = subparsers.add_parser("process", help="Process and prepare data")
        process_parser.add_argument("--country", required=True, help="Country code")
        process_parser.add_argument("--input-file", required=True, help="Input data file")
        process_parser.add_argument("--output-file", required=True, help="Output processed file")
        
        # Convert data
        convert_parser = subparsers.add_parser("convert", help="Convert CSV to JSONL format")
        convert_parser.add_argument("--input-path", required=True, help="Input CSV file path")
        convert_parser.add_argument("--output-path", help="Output JSONL file path (not required when using -ov)")
        convert_parser.add_argument("--country", required=True, help="Country code for configuration")
        convert_parser.add_argument("--text-column", default="formatted_address", help="Text column name (default: formatted_address)")
        convert_parser.add_argument("--validation-mode", choices=["strict", "lenient"], default="strict", help="Validation mode (default: strict)")
        convert_parser.add_argument("-ov", "--only-validate", action="store_true", help="Only validate CSV data without conversion")
        convert_parser.add_argument("-fix", "--fix", action="store_true", help="Automatically process anomalies (equivalent to auto_process=True)")
        
        # Split data
        split_parser = subparsers.add_parser("split", help="Split data into train/val/test")
        split_parser.add_argument("--input-file", required=True, help="Input data file")
        split_parser.add_argument("--train-ratio", type=float, default=0.8, help="Training data ratio")
        split_parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation data ratio")
        split_parser.add_argument("--test-ratio", type=float, default=0.1, help="Test data ratio")
        split_parser.add_argument("--output-dir", required=True, help="Output directory")
    
    def execute(self, args) -> bool:
        try:
            from ..data import NERDataProcessor
            
            if args.data_action == "validate":
                config = self.get_country_config(args.country)
                
                processor = NERDataProcessor(config, logger=self.logger)
                is_valid = processor.validate_data_file(args.input_file)
                
                if is_valid:
                    self.logger.info(f"Data file '{args.input_file}' is valid")
                else:
                    self.logger.error(f"Data file '{args.input_file}' has validation errors")
                    return False
                
            elif args.data_action == "process":
                config = self.get_country_config(args.country)
                
                processor = NERDataProcessor(config, logger=self.logger)
                processor.process_file(args.input_file, args.output_file)
                self.logger.info(f"Processed data saved to '{args.output_file}'")
                
            elif args.data_action == "convert":
                from ..data.convertor import CSVAnnotationConvert
                
                # Initialize CSV annotation generator
                generator = CSVAnnotationConvert(config_dir=self.global_config.get('config_dir', 'data/ner/configs'))
                
                # Check if only validation is requested
                if getattr(args, 'only_validate', False):
                    try:
                        validation_report = generator.validate_csv(
                            csv_path=args.input_path,
                            country_code=args.country,
                            text_column=args.text_column,
                            auto_process=getattr(args, 'fix', False)
                        )
                        
                        # Generate validation report CSV
                        csv_path = Path(args.input_path)
                        report_csv_path = csv_path.parent / f"{csv_path.stem}_validation_report.csv"
                        generator.print_validation_report(validation_report, str(report_csv_path), show_details=True)
                        
                        self.logger.info(f"Validation completed. Success rate: {validation_report.success_rate:.2%}")
                        self.logger.info(f"Total rows: {validation_report.total_rows}, Valid rows: {validation_report.valid_rows}")
                        self.logger.info(f"Validation report saved to: {report_csv_path}")
                        
                        if validation_report.success_rate < 1.0:
                            self.logger.warning(f"Found {len(validation_report.issues)} data quality issues")
                            if args.validation_mode == "strict":
                                self.logger.error("Validation failed in strict mode. Use --validation-mode lenient to continue with issues.")
                                return False
                        else:
                            self.logger.info("All data passed validation successfully!")
                            
                    except Exception as e:
                        self.logger.error(f"CSV validation failed: {e}")
                        return False
                        
                else:
                    if not getattr(args, 'output_path', None):
                        self.logger.error("--output-path is required when not using -ov (only validate) mode")
                        return False
                        
                    try:
                        output_path = generator.generate_from_csv(
                            csv_path=args.input_path,
                            country_code=args.country,
                            output_file=args.output_path,
                            text_column=args.text_column,
                            validate_text_contains_entities=True,
                            validation_mode=args.validation_mode
                        )
                        
                        self.logger.info(f"Successfully converted CSV to JSONL: {output_path}")
                        
                    except ValueError as e:
                        self.logger.error(f"CSV conversion failed: {e}")
                        if args.validation_mode == "strict":
                            self.logger.error("Use --validation-mode lenient to continue with data quality issues")
                        return False
                
            elif args.data_action == "split":
                raise NotImplementedError("Data split not implemented")

            else:
                self.logger.error("Please specify a data action")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Data operation failed: {e}")
            return False
