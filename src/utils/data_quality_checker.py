"""
数据质量检查工具
用于检查和修正NER训练数据中的标注错误
"""

import json
from typing import List, Dict, Tuple

class DataQualityChecker:
    def __init__(self):
        # BIO标签验证规则
        self.valid_bio_prefixes = ['B-', 'I-', 'O']
        self.entity_types = ['STREET', 'CITY', 'DISTRICT', 'BUILDING', 'COUNTRY', 'POSTAL_CODE']
        
    def check_bio_consistency(self, labels: List[str]) -> List[str]:
        """检查BIO标签的一致性"""
        errors = []
        
        for i, label in enumerate(labels):
            if label == 'O':
                continue
                
            # 检查I-标签前是否有对应的B-标签
            if label.startswith('I-'):
                entity_type = label[2:]
                
                # 检查前一个标签
                if i == 0:
                    errors.append(f"位置 {i}: I-{entity_type} 出现在开头，应该是 B-{entity_type}")
                else:
                    prev_label = labels[i-1]
                    if prev_label == 'O':
                        errors.append(f"位置 {i}: I-{entity_type} 前面是 O，应该是 B-{entity_type}")
                    elif prev_label.startswith('B-') or prev_label.startswith('I-'):
                        prev_entity = prev_label[2:]
                        if prev_entity != entity_type:
                            errors.append(f"位置 {i}: I-{entity_type} 与前面的 {prev_label} 类型不匹配")
        
        return errors
    
    def suggest_corrections(self, tokens: List[str], labels: List[str]) -> List[str]:
        """基于常见模式建议标签修正"""
        corrected_labels = labels.copy()
        
        for i, (token, label) in enumerate(zip(tokens, labels)):
            # 街道关键词
            if token in ['شارع', 'طريق', 'شارع']:
                if label == 'O':
                    corrected_labels[i] = 'B-STREET'
                    # 标记后续相关词汇
                    for j in range(i+1, min(i+4, len(tokens))):
                        if corrected_labels[j] == 'O' and not tokens[j].endswith('،'):
                            corrected_labels[j] = 'I-STREET'
                        else:
                            break
            
            # 建筑物关键词
            elif token in ['مبنى', 'رقم']:
                if label == 'O':
                    corrected_labels[i] = 'B-BUILDING'
                    # 标记后续数字
                    if i+1 < len(tokens) and tokens[i+1].isdigit():
                        corrected_labels[i+1] = 'I-BUILDING'
            
            # 区域关键词
            elif token in ['حي', 'منطقة', 'المنطقة']:
                if label == 'O':
                    corrected_labels[i] = 'B-DISTRICT'
                    # 标记后续词汇
                    if i+1 < len(tokens) and corrected_labels[i+1] == 'O':
                        corrected_labels[i+1] = 'I-DISTRICT'
        
        return corrected_labels
    
    def analyze_dataset(self, data_path: str) -> Dict:
        """分析整个数据集的质量"""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        analysis = {
            'total_samples': len(data),
            'bio_errors': [],
            'label_distribution': {},
            'suggestions': []
        }
        
        for i, sample in enumerate(data):
            tokens = sample['tokens']
            labels = sample['labels']
            
            # 检查BIO一致性
            bio_errors = self.check_bio_consistency(labels)
            if bio_errors:
                analysis['bio_errors'].append({
                    'sample_index': i,
                    'text': sample['original_text'],
                    'errors': bio_errors
                })
            
            # 统计标签分布
            for label in labels:
                analysis['label_distribution'][label] = analysis['label_distribution'].get(label, 0) + 1
            
            # 生成修正建议
            suggested_labels = self.suggest_corrections(tokens, labels)
            if suggested_labels != labels:
                analysis['suggestions'].append({
                    'sample_index': i,
                    'text': sample['original_text'],
                    'original_labels': labels,
                    'suggested_labels': suggested_labels
                })
        
        return analysis
    
    def print_analysis_report(self, analysis: Dict):
        """打印分析报告"""
        print("=== 数据质量分析报告 ===\n")
        
        print(f"📊 总样本数: {analysis['total_samples']}")
        print(f"❌ BIO错误数: {len(analysis['bio_errors'])}")
        print(f"💡 修正建议数: {len(analysis['suggestions'])}\n")
        
        print("📈 标签分布:")
        for label, count in sorted(analysis['label_distribution'].items()):
            print(f"  {label}: {count}")
        print()
        
        if analysis['bio_errors']:
            print("❌ BIO标签错误:")
            for error in analysis['bio_errors'][:3]:  # 只显示前3个
                print(f"  样本 {error['sample_index']}: {error['text']}")
                for err in error['errors']:
                    print(f"    - {err}")
            print()
        
        if analysis['suggestions']:
            print("💡 修正建议 (前3个):")
            for suggestion in analysis['suggestions'][:3]:
                print(f"  样本 {suggestion['sample_index']}: {suggestion['text']}")
                print(f"    原标签: {suggestion['original_labels']}")
                print(f"    建议标签: {suggestion['suggested_labels']}")
                print()

# 使用示例
if __name__ == "__main__":
    checker = DataQualityChecker()
    
    # 分析原始数据
    print("分析原始数据...")
    analysis = checker.analyze_dataset('/Users/zexho/Documents/python_script/araBertv2/data/processed/processed_sample.json')
    checker.print_analysis_report(analysis)
    
    print("\n" + "="*50 + "\n")
    
    # 分析修正后的数据
    print("分析修正后的数据...")
    analysis_corrected = checker.analyze_dataset('/Users/zexho/Documents/python_script/araBertv2/data/processed/corrected_sample.json')
    checker.print_analysis_report(analysis_corrected)