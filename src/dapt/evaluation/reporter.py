"""DAPT Evaluation Reporter

Report generation for DAPT training evaluation:
- Generate comprehensive evaluation reports
- Create visualizations and charts
- Export reports in multiple formats
- Compare model performances
- Track evaluation history
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, asdict
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_agg import FigureCanvasAgg
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from jinja2 import Template

@dataclass
class ReportConfig:
    """Configuration for report generation"""
    title: str
    subtitle: str = ""
    author: str = "DAPT Training System"
    include_plots: bool = True
    include_raw_data: bool = False
    plot_style: str = "seaborn"
    color_palette: str = "viridis"
    output_formats: List[str] = None
    template_name: str = "default"
    
    def __post_init__(self):
        if self.output_formats is None:
            self.output_formats = ['html', 'pdf']

class PlotGenerator:
    """Generate plots for evaluation reports"""
    
    def __init__(self, style: str = "seaborn", palette: str = "viridis"):
        self.style = style
        self.palette = palette
        
        # Set matplotlib style
        plt.style.use(self.style if self.style in plt.style.available else 'default')
        sns.set_palette(self.palette)
    
    def create_metrics_comparison_plot(self, comparison_data: Dict[str, Dict[str, float]], 
                                     metric_name: str = "f1_score") -> str:
        """Create metrics comparison bar plot"""
        try:
            models = list(comparison_data.keys())
            values = [data.get(metric_name, 0.0) for data in comparison_data.values()]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(models, values, color=sns.color_palette(self.palette, len(models)))
            
            ax.set_title(f'Model Comparison - {metric_name.replace("_", " ").title()}', fontsize=16, fontweight='bold')
            ax.set_xlabel('Models', fontsize=12)
            ax.set_ylabel(metric_name.replace("_", " ").title(), fontsize=12)
            ax.set_ylim(0, max(values) * 1.1 if values else 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # Rotate x-axis labels if needed
            if len(max(models, key=len)) > 10:
                plt.xticks(rotation=45, ha='right')
            
            plt.tight_layout()
            
            # Convert to base64 string
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return f"data:image/png;base64,{plot_data}"
            
        except Exception as e:
            logging.error(f"Error creating metrics comparison plot: {str(e)}")
            return ""
    
    def create_metrics_heatmap(self, metrics_data: Dict[str, Dict[str, float]]) -> str:
        """Create heatmap of metrics across models"""
        try:
            # Convert to DataFrame
            df = pd.DataFrame(metrics_data).T
            
            if df.empty:
                return ""
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create heatmap
            sns.heatmap(df, annot=True, cmap=self.palette, fmt='.3f', 
                       cbar_kws={'label': 'Metric Value'}, ax=ax)
            
            ax.set_title('Metrics Heatmap Across Models', fontsize=16, fontweight='bold')
            ax.set_xlabel('Metrics', fontsize=12)
            ax.set_ylabel('Models', fontsize=12)
            
            plt.tight_layout()
            
            # Convert to base64 string
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return f"data:image/png;base64,{plot_data}"
            
        except Exception as e:
            logging.error(f"Error creating metrics heatmap: {str(e)}")
            return ""
    
    def create_training_progress_plot(self, training_history: List[Dict[str, Any]]) -> str:
        """Create training progress plot"""
        try:
            if not training_history:
                return ""
            
            # Extract data
            epochs = []
            train_loss = []
            val_loss = []
            
            for entry in training_history:
                epochs.append(entry.get('epoch', 0))
                train_loss.append(entry.get('train_loss', 0))
                val_loss.append(entry.get('val_loss', 0))
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.plot(epochs, train_loss, label='Training Loss', marker='o', linewidth=2)
            ax.plot(epochs, val_loss, label='Validation Loss', marker='s', linewidth=2)
            
            ax.set_title('Training Progress', fontsize=16, fontweight='bold')
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Loss', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Convert to base64 string
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return f"data:image/png;base64,{plot_data}"
            
        except Exception as e:
            logging.error(f"Error creating training progress plot: {str(e)}")
            return ""
    
    def create_confusion_matrix_plot(self, confusion_matrix: List[List[int]], 
                                   labels: List[str]) -> str:
        """Create confusion matrix heatmap"""
        try:
            if not confusion_matrix or not labels:
                return ""
            
            cm = np.array(confusion_matrix)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=labels, yticklabels=labels, ax=ax)
            
            ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold')
            ax.set_xlabel('Predicted Label', fontsize=12)
            ax.set_ylabel('True Label', fontsize=12)
            
            plt.tight_layout()
            
            # Convert to base64 string
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return f"data:image/png;base64,{plot_data}"
            
        except Exception as e:
            logging.error(f"Error creating confusion matrix plot: {str(e)}")
            return ""
    
    def create_performance_timeline(self, evaluation_history: List[Dict[str, Any]]) -> str:
        """Create performance timeline plot"""
        try:
            if not evaluation_history:
                return ""
            
            # Extract data
            dates = []
            f1_scores = []
            models = []
            
            for entry in evaluation_history:
                try:
                    date = datetime.fromisoformat(entry.get('evaluation_time', ''))
                    dates.append(date)
                    f1_scores.append(entry.get('metrics', {}).get('f1_score', 0))
                    models.append(entry.get('model_name', 'Unknown'))
                except:
                    continue
            
            if not dates:
                return ""
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Create scatter plot with different colors for different models
            unique_models = list(set(models))
            colors = sns.color_palette(self.palette, len(unique_models))
            
            for i, model in enumerate(unique_models):
                model_dates = [d for d, m in zip(dates, models) if m == model]
                model_scores = [s for s, m in zip(f1_scores, models) if m == model]
                
                ax.scatter(model_dates, model_scores, label=model, 
                          color=colors[i], s=60, alpha=0.7)
                ax.plot(model_dates, model_scores, color=colors[i], alpha=0.5)
            
            ax.set_title('Model Performance Timeline', fontsize=16, fontweight='bold')
            ax.set_xlabel('Evaluation Date', fontsize=12)
            ax.set_ylabel('F1 Score', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Format x-axis
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            
            # Convert to base64 string
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return f"data:image/png;base64,{plot_data}"
            
        except Exception as e:
            logging.error(f"Error creating performance timeline: {str(e)}")
            return ""

class EvaluationReporter:
    """Main evaluation reporter for DAPT training"""
    
    def __init__(self, config: Dict[str, Any], global_config: Dict[str, Any]):
        self.config = config
        self.global_config = global_config
        self.country_code = config["country"]["code"]
        
        # Report paths
        self.reports_dir = Path(global_config["reports_dir"]) / self.country_code
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        self.templates_dir = Path(__file__).parent / "templates"
        self.templates_dir.mkdir(exist_ok=True)
        
        # Plot generator
        self.plot_generator = PlotGenerator()
        
        # Logging
        self.logger = self._setup_logging()
        
        # Create default templates if they don't exist
        self._create_default_templates()
        
        self.logger.info(f"Evaluation Reporter initialized for country: {self.country_code}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for evaluation reporter"""
        logger = logging.getLogger(f"dapt_reporter_{self.country_code}")
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
    
    def generate_evaluation_report(self, evaluation_results: List[Dict[str, Any]],
                                 report_config: ReportConfig,
                                 comparison_data: Optional[Dict[str, Any]] = None,
                                 training_history: Optional[List[Dict[str, Any]]] = None) -> Optional[str]:
        """Generate comprehensive evaluation report"""
        try:
            if not evaluation_results:
                self.logger.warning("No evaluation results provided for report generation")
                return None
            
            self.logger.info(f"Generating evaluation report: {report_config.title}")
            
            # Prepare report data
            report_data = self._prepare_report_data(
                evaluation_results, comparison_data, training_history
            )
            
            # Generate plots if requested
            plots = {}
            if report_config.include_plots:
                plots = self._generate_plots(report_data)
            
            # Create report content
            report_content = self._create_report_content(
                report_data, plots, report_config
            )
            
            # Generate output files
            output_files = []
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            for output_format in report_config.output_formats:
                output_file = self._generate_output_file(
                    report_content, output_format, report_config, timestamp
                )
                if output_file:
                    output_files.append(output_file)
            
            if output_files:
                self.logger.info(f"Report generated successfully: {output_files}")
                return output_files[0]  # Return primary output file
            else:
                self.logger.error("Failed to generate any output files")
                return None
                
        except Exception as e:
            self.logger.error(f"Error generating evaluation report: {str(e)}")
            return None
    
    def _prepare_report_data(self, evaluation_results: List[Dict[str, Any]],
                           comparison_data: Optional[Dict[str, Any]],
                           training_history: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Prepare data for report generation"""
        try:
            # Process evaluation results
            processed_results = []
            for result in evaluation_results:
                processed_result = {
                    'model_name': result.get('model_name', 'Unknown'),
                    'dataset_name': result.get('dataset_name', 'Unknown'),
                    'evaluation_time': result.get('evaluation_time', ''),
                    'metrics': result.get('metrics', {}),
                    'detailed_metrics': result.get('detailed_metrics', {}),
                    'task_type': result.get('evaluation_config', {}).get('task_type', 'unknown')
                }
                processed_results.append(processed_result)
            
            # Calculate summary statistics
            summary_stats = self._calculate_summary_statistics(processed_results)
            
            # Prepare model comparison data
            model_comparison = self._prepare_model_comparison(processed_results)
            
            # Prepare dataset analysis
            dataset_analysis = self._prepare_dataset_analysis(processed_results)
            
            return {
                'evaluation_results': processed_results,
                'summary_statistics': summary_stats,
                'model_comparison': model_comparison,
                'dataset_analysis': dataset_analysis,
                'comparison_data': comparison_data or {},
                'training_history': training_history or [],
                'country_code': self.country_code,
                'generation_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error preparing report data: {str(e)}")
            return {}
    
    def _calculate_summary_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate summary statistics from evaluation results"""
        try:
            if not results:
                return {}
            
            # Extract metrics
            all_metrics = {}
            for result in results:
                for metric_name, value in result.get('metrics', {}).items():
                    if metric_name not in all_metrics:
                        all_metrics[metric_name] = []
                    all_metrics[metric_name].append(value)
            
            # Calculate statistics for each metric
            stats = {}
            for metric_name, values in all_metrics.items():
                if values:
                    stats[metric_name] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'median': float(np.median(values)),
                        'count': len(values)
                    }
            
            # Overall statistics
            total_evaluations = len(results)
            unique_models = len(set(r.get('model_name') for r in results))
            unique_datasets = len(set(r.get('dataset_name') for r in results))
            
            return {
                'metric_statistics': stats,
                'total_evaluations': total_evaluations,
                'unique_models': unique_models,
                'unique_datasets': unique_datasets,
                'evaluation_period': self._get_evaluation_period(results)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating summary statistics: {str(e)}")
            return {}
    
    def _prepare_model_comparison(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare model comparison data"""
        try:
            model_data = {}
            
            for result in results:
                model_name = result.get('model_name', 'Unknown')
                if model_name not in model_data:
                    model_data[model_name] = {
                        'evaluations': [],
                        'average_metrics': {},
                        'best_performance': {},
                        'datasets_evaluated': set()
                    }
                
                model_data[model_name]['evaluations'].append(result)
                model_data[model_name]['datasets_evaluated'].add(result.get('dataset_name', 'Unknown'))
            
            # Calculate averages and best performance for each model
            for model_name, data in model_data.items():
                # Convert set to list for JSON serialization
                data['datasets_evaluated'] = list(data['datasets_evaluated'])
                
                # Calculate average metrics
                all_metrics = {}
                for evaluation in data['evaluations']:
                    for metric_name, value in evaluation.get('metrics', {}).items():
                        if metric_name not in all_metrics:
                            all_metrics[metric_name] = []
                        all_metrics[metric_name].append(value)
                
                for metric_name, values in all_metrics.items():
                    data['average_metrics'][metric_name] = float(np.mean(values))
                
                # Find best performance
                if data['evaluations']:
                    best_eval = max(data['evaluations'], 
                                  key=lambda x: x.get('metrics', {}).get('f1_score', 0))
                    data['best_performance'] = best_eval.get('metrics', {})
            
            return model_data
            
        except Exception as e:
            self.logger.error(f"Error preparing model comparison: {str(e)}")
            return {}
    
    def _prepare_dataset_analysis(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare dataset analysis data"""
        try:
            dataset_data = {}
            
            for result in results:
                dataset_name = result.get('dataset_name', 'Unknown')
                if dataset_name not in dataset_data:
                    dataset_data[dataset_name] = {
                        'evaluations': [],
                        'models_tested': set(),
                        'performance_range': {},
                        'task_type': result.get('task_type', 'unknown')
                    }
                
                dataset_data[dataset_name]['evaluations'].append(result)
                dataset_data[dataset_name]['models_tested'].add(result.get('model_name', 'Unknown'))
            
            # Calculate performance ranges for each dataset
            for dataset_name, data in dataset_data.items():
                # Convert set to list for JSON serialization
                data['models_tested'] = list(data['models_tested'])
                
                # Calculate performance ranges
                all_metrics = {}
                for evaluation in data['evaluations']:
                    for metric_name, value in evaluation.get('metrics', {}).items():
                        if metric_name not in all_metrics:
                            all_metrics[metric_name] = []
                        all_metrics[metric_name].append(value)
                
                for metric_name, values in all_metrics.items():
                    if values:
                        data['performance_range'][metric_name] = {
                            'min': float(np.min(values)),
                            'max': float(np.max(values)),
                            'range': float(np.max(values) - np.min(values))
                        }
            
            return dataset_data
            
        except Exception as e:
            self.logger.error(f"Error preparing dataset analysis: {str(e)}")
            return {}
    
    def _get_evaluation_period(self, results: List[Dict[str, Any]]) -> Dict[str, str]:
        """Get evaluation period from results"""
        try:
            dates = []
            for result in results:
                eval_time = result.get('evaluation_time', '')
                if eval_time:
                    try:
                        dates.append(datetime.fromisoformat(eval_time))
                    except:
                        continue
            
            if dates:
                return {
                    'start': min(dates).isoformat(),
                    'end': max(dates).isoformat(),
                    'duration_days': (max(dates) - min(dates)).days
                }
            else:
                return {'start': '', 'end': '', 'duration_days': 0}
                
        except Exception:
            return {'start': '', 'end': '', 'duration_days': 0}
    
    def _generate_plots(self, report_data: Dict[str, Any]) -> Dict[str, str]:
        """Generate plots for the report"""
        try:
            plots = {}
            
            # Model comparison plot
            model_comparison = report_data.get('model_comparison', {})
            if model_comparison:
                comparison_metrics = {}
                for model_name, data in model_comparison.items():
                    comparison_metrics[model_name] = data.get('average_metrics', {})
                
                if comparison_metrics:
                    plots['model_comparison'] = self.plot_generator.create_metrics_comparison_plot(
                        comparison_metrics, 'f1_score'
                    )
                    plots['metrics_heatmap'] = self.plot_generator.create_metrics_heatmap(
                        comparison_metrics
                    )
            
            # Training progress plot
            training_history = report_data.get('training_history', [])
            if training_history:
                plots['training_progress'] = self.plot_generator.create_training_progress_plot(
                    training_history
                )
            
            # Performance timeline
            evaluation_results = report_data.get('evaluation_results', [])
            if evaluation_results:
                plots['performance_timeline'] = self.plot_generator.create_performance_timeline(
                    evaluation_results
                )
            
            # Confusion matrix (if available)
            for result in evaluation_results:
                detailed_metrics = result.get('detailed_metrics', {})
                if 'confusion_matrix' in detailed_metrics and 'label_names' in detailed_metrics:
                    plots['confusion_matrix'] = self.plot_generator.create_confusion_matrix_plot(
                        detailed_metrics['confusion_matrix'],
                        detailed_metrics['label_names']
                    )
                    break  # Only create one confusion matrix plot
            
            return plots
            
        except Exception as e:
            self.logger.error(f"Error generating plots: {str(e)}")
            return {}
    
    def _create_report_content(self, report_data: Dict[str, Any], 
                             plots: Dict[str, str],
                             report_config: ReportConfig) -> Dict[str, Any]:
        """Create report content structure"""
        try:
            content = {
                'title': report_config.title,
                'subtitle': report_config.subtitle,
                'author': report_config.author,
                'generation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'country_code': self.country_code,
                'report_data': report_data,
                'plots': plots,
                'include_plots': report_config.include_plots,
                'include_raw_data': report_config.include_raw_data
            }
            
            return content
            
        except Exception as e:
            self.logger.error(f"Error creating report content: {str(e)}")
            return {}
    
    def _generate_output_file(self, content: Dict[str, Any], 
                            output_format: str,
                            report_config: ReportConfig,
                            timestamp: str) -> Optional[str]:
        """Generate output file in specified format"""
        try:
            filename = f"{report_config.title.lower().replace(' ', '_')}_{timestamp}"
            
            if output_format.lower() == 'html':
                return self._generate_html_report(content, filename)
            elif output_format.lower() == 'pdf':
                return self._generate_pdf_report(content, filename)
            elif output_format.lower() == 'json':
                return self._generate_json_report(content, filename)
            elif output_format.lower() == 'markdown':
                return self._generate_markdown_report(content, filename)
            else:
                self.logger.warning(f"Unsupported output format: {output_format}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error generating {output_format} output: {str(e)}")
            return None
    
    def _generate_html_report(self, content: Dict[str, Any], filename: str) -> Optional[str]:
        """Generate HTML report"""
        try:
            template_path = self.templates_dir / "report_template.html"
            
            with open(template_path, 'r', encoding='utf-8') as f:
                template_content = f.read()
            
            template = Template(template_content)
            html_content = template.render(**content)
            
            output_path = self.reports_dir / f"{filename}.html"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Error generating HTML report: {str(e)}")
            return None
    
    def _generate_pdf_report(self, content: Dict[str, Any], filename: str) -> Optional[str]:
        """Generate PDF report"""
        try:
            # First generate HTML
            html_file = self._generate_html_report(content, f"{filename}_temp")
            if not html_file:
                return None
            
            # Convert HTML to PDF
            output_path = self.reports_dir / f"{filename}.pdf"
            
            try:
                HTML(filename=html_file).write_pdf(str(output_path))
                
                # Clean up temporary HTML file
                os.remove(html_file)
                
                return str(output_path)
                
            except Exception as e:
                self.logger.warning(f"PDF generation failed with weasyprint: {str(e)}")
                # Fallback: just return the HTML file
                return html_file
            
        except Exception as e:
            self.logger.error(f"Error generating PDF report: {str(e)}")
            return None
    
    def _generate_json_report(self, content: Dict[str, Any], filename: str) -> Optional[str]:
        """Generate JSON report"""
        try:
            output_path = self.reports_dir / f"{filename}.json"
            
            # Remove plots from JSON (they're base64 encoded and too large)
            json_content = content.copy()
            json_content['plots'] = {k: "[Plot data removed for JSON export]" 
                                   for k in content.get('plots', {}).keys()}
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_content, f, indent=2, ensure_ascii=False, default=str)
            
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Error generating JSON report: {str(e)}")
            return None
    
    def _generate_markdown_report(self, content: Dict[str, Any], filename: str) -> Optional[str]:
        """Generate Markdown report"""
        try:
            template_path = self.templates_dir / "report_template.md"
            
            with open(template_path, 'r', encoding='utf-8') as f:
                template_content = f.read()
            
            template = Template(template_content)
            markdown_content = template.render(**content)
            
            output_path = self.reports_dir / f"{filename}.md"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Error generating Markdown report: {str(e)}")
            return None
    
    def _create_default_templates(self) -> None:
        """Create default report templates"""
        try:
            # HTML template
            html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            border-bottom: 3px solid #007acc;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        .header h1 {
            color: #007acc;
            margin: 0;
            font-size: 2.5em;
        }
        .header h2 {
            color: #666;
            margin: 10px 0;
            font-weight: normal;
        }
        .section {
            margin: 30px 0;
        }
        .section h2 {
            color: #007acc;
            border-bottom: 2px solid #007acc;
            padding-bottom: 10px;
        }
        .metrics-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        .metrics-table th,
        .metrics-table td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        .metrics-table th {
            background-color: #007acc;
            color: white;
        }
        .metrics-table tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .plot-container {
            text-align: center;
            margin: 20px 0;
        }
        .plot-container img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .summary-card {
            background-color: #f8f9fa;
            border-left: 4px solid #007acc;
            padding: 20px;
            margin: 20px 0;
        }
        .metric-value {
            font-weight: bold;
            color: #007acc;
        }
        .footer {
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{{ title }}</h1>
            {% if subtitle %}<h2>{{ subtitle }}</h2>{% endif %}
            <p><strong>Country:</strong> {{ country_code }} | <strong>Generated:</strong> {{ generation_date }} | <strong>Author:</strong> {{ author }}</p>
        </div>

        <!-- Summary Section -->
        <div class="section">
            <h2>Executive Summary</h2>
            <div class="summary-card">
                <p><strong>Total Evaluations:</strong> <span class="metric-value">{{ report_data.summary_statistics.total_evaluations }}</span></p>
                <p><strong>Models Evaluated:</strong> <span class="metric-value">{{ report_data.summary_statistics.unique_models }}</span></p>
                <p><strong>Datasets Used:</strong> <span class="metric-value">{{ report_data.summary_statistics.unique_datasets }}</span></p>
                {% if report_data.summary_statistics.evaluation_period.duration_days > 0 %}
                <p><strong>Evaluation Period:</strong> {{ report_data.summary_statistics.evaluation_period.duration_days }} days</p>
                {% endif %}
            </div>
        </div>

        <!-- Model Comparison Section -->
        {% if report_data.model_comparison %}
        <div class="section">
            <h2>Model Performance Comparison</h2>
            {% if include_plots and plots.model_comparison %}
            <div class="plot-container">
                <img src="{{ plots.model_comparison }}" alt="Model Comparison Plot">
            </div>
            {% endif %}
            
            <table class="metrics-table">
                <thead>
                    <tr>
                        <th>Model</th>
                        <th>Datasets Evaluated</th>
                        <th>Avg F1 Score</th>
                        <th>Avg Accuracy</th>
                        <th>Avg Precision</th>
                        <th>Avg Recall</th>
                    </tr>
                </thead>
                <tbody>
                    {% for model_name, data in report_data.model_comparison.items() %}
                    <tr>
                        <td>{{ model_name }}</td>
                        <td>{{ data.datasets_evaluated|length }}</td>
                        <td class="metric-value">{{ "%.4f"|format(data.average_metrics.get('f1_score', 0)) }}</td>
                        <td class="metric-value">{{ "%.4f"|format(data.average_metrics.get('accuracy', 0)) }}</td>
                        <td class="metric-value">{{ "%.4f"|format(data.average_metrics.get('precision', 0)) }}</td>
                        <td class="metric-value">{{ "%.4f"|format(data.average_metrics.get('recall', 0)) }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        {% endif %}

        <!-- Metrics Heatmap -->
        {% if include_plots and plots.metrics_heatmap %}
        <div class="section">
            <h2>Metrics Overview</h2>
            <div class="plot-container">
                <img src="{{ plots.metrics_heatmap }}" alt="Metrics Heatmap">
            </div>
        </div>
        {% endif %}

        <!-- Performance Timeline -->
        {% if include_plots and plots.performance_timeline %}
        <div class="section">
            <h2>Performance Timeline</h2>
            <div class="plot-container">
                <img src="{{ plots.performance_timeline }}" alt="Performance Timeline">
            </div>
        </div>
        {% endif %}

        <!-- Training Progress -->
        {% if include_plots and plots.training_progress %}
        <div class="section">
            <h2>Training Progress</h2>
            <div class="plot-container">
                <img src="{{ plots.training_progress }}" alt="Training Progress">
            </div>
        </div>
        {% endif %}

        <!-- Confusion Matrix -->
        {% if include_plots and plots.confusion_matrix %}
        <div class="section">
            <h2>Confusion Matrix</h2>
            <div class="plot-container">
                <img src="{{ plots.confusion_matrix }}" alt="Confusion Matrix">
            </div>
        </div>
        {% endif %}

        <!-- Detailed Results -->
        {% if include_raw_data %}
        <div class="section">
            <h2>Detailed Evaluation Results</h2>
            {% for result in report_data.evaluation_results %}
            <div class="summary-card">
                <h3>{{ result.model_name }} on {{ result.dataset_name }}</h3>
                <p><strong>Task Type:</strong> {{ result.task_type }}</p>
                <p><strong>Evaluation Time:</strong> {{ result.evaluation_time }}</p>
                <table class="metrics-table">
                    <thead>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for metric_name, value in result.metrics.items() %}
                        <tr>
                            <td>{{ metric_name.replace('_', ' ').title() }}</td>
                            <td class="metric-value">{{ "%.4f"|format(value) }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
            {% endfor %}
        </div>
        {% endif %}

        <div class="footer">
            <p>Generated by DAPT Training System | {{ generation_date }}</p>
        </div>
    </div>
</body>
</html>
"""
            
            html_template_path = self.templates_dir / "report_template.html"
            with open(html_template_path, 'w', encoding='utf-8') as f:
                f.write(html_template)
            
            # Markdown template
            markdown_template = """
# {{ title }}

{% if subtitle %}## {{ subtitle }}{% endif %}

**Country:** {{ country_code }}  
**Generated:** {{ generation_date }}  
**Author:** {{ author }}

## Executive Summary

- **Total Evaluations:** {{ report_data.summary_statistics.total_evaluations }}
- **Models Evaluated:** {{ report_data.summary_statistics.unique_models }}
- **Datasets Used:** {{ report_data.summary_statistics.unique_datasets }}
{% if report_data.summary_statistics.evaluation_period.duration_days > 0 %}
- **Evaluation Period:** {{ report_data.summary_statistics.evaluation_period.duration_days }} days
{% endif %}

## Model Performance Comparison

{% if report_data.model_comparison %}
| Model | Datasets | Avg F1 | Avg Accuracy | Avg Precision | Avg Recall |
|-------|----------|--------|--------------|---------------|------------|
{% for model_name, data in report_data.model_comparison.items() %}
| {{ model_name }} | {{ data.datasets_evaluated|length }} | {{ "%.4f"|format(data.average_metrics.get('f1_score', 0)) }} | {{ "%.4f"|format(data.average_metrics.get('accuracy', 0)) }} | {{ "%.4f"|format(data.average_metrics.get('precision', 0)) }} | {{ "%.4f"|format(data.average_metrics.get('recall', 0)) }} |
{% endfor %}
{% endif %}

{% if include_raw_data %}
## Detailed Evaluation Results

{% for result in report_data.evaluation_results %}
### {{ result.model_name }} on {{ result.dataset_name }}

- **Task Type:** {{ result.task_type }}
- **Evaluation Time:** {{ result.evaluation_time }}

**Metrics:**
{% for metric_name, value in result.metrics.items() %}
- **{{ metric_name.replace('_', ' ').title() }}:** {{ "%.4f"|format(value) }}
{% endfor %}

{% endfor %}
{% endif %}

---
*Generated by DAPT Training System | {{ generation_date }}*
"""
            
            markdown_template_path = self.templates_dir / "report_template.md"
            with open(markdown_template_path, 'w', encoding='utf-8') as f:
                f.write(markdown_template)
            
            self.logger.debug("Default report templates created")
            
        except Exception as e:
            self.logger.error(f"Error creating default templates: {str(e)}")
    
    def create_comparison_report(self, baseline_results: Dict[str, Any],
                               comparison_results: List[Dict[str, Any]],
                               report_title: str = "Model Comparison Report") -> Optional[str]:
        """Create a comparison report between baseline and other models"""
        try:
            # Prepare comparison data
            all_results = [baseline_results] + comparison_results
            
            # Create report config
            report_config = ReportConfig(
                title=report_title,
                subtitle=f"Baseline vs {len(comparison_results)} Comparison Models",
                include_plots=True,
                include_raw_data=True
            )
            
            # Generate report
            return self.generate_evaluation_report(
                all_results, report_config
            )
            
        except Exception as e:
            self.logger.error(f"Error creating comparison report: {str(e)}")
            return None
    
    def get_report_summary(self) -> Dict[str, Any]:
        """Get summary of generated reports"""
        try:
            reports = list(self.reports_dir.glob("*.html"))
            reports.extend(self.reports_dir.glob("*.pdf"))
            reports.extend(self.reports_dir.glob("*.json"))
            reports.extend(self.reports_dir.glob("*.md"))
            
            report_info = []
            for report_path in reports:
                try:
                    stat = report_path.stat()
                    report_info.append({
                        'name': report_path.name,
                        'path': str(report_path),
                        'size_bytes': stat.st_size,
                        'created_time': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                        'format': report_path.suffix[1:]
                    })
                except Exception:
                    continue
            
            # Sort by creation time (newest first)
            report_info.sort(key=lambda x: x['created_time'], reverse=True)
            
            return {
                'total_reports': len(report_info),
                'reports_directory': str(self.reports_dir),
                'reports': report_info,
                'formats_available': list(set(r['format'] for r in report_info))
            }
            
        except Exception as e:
            self.logger.error(f"Error getting report summary: {str(e)}")
            return {}