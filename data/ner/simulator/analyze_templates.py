#!/usr/bin/env python3
"""
地址模板分析工具

分析标注数据CSV文件，统计各种实体组合模板的出现频率，
帮助生成符合真实情况的地址模板配置。

使用方法：
    python3 data/ner/simulator/analyze_templates.py <csv_file_path>
    
示例：
    python3 data/ner/simulator/analyze_templates.py \
        data/ner/raw_data/第三次训练/线上数据-第三次训练-01-标注结果.csv
"""

import csv
import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Tuple


class TemplateAnalyzer:
    """地址模板分析器"""
    
    # 实体类型及其显示顺序（从小到大）
    ENTITY_TYPES = ['BUILDING', 'STREET', 'COMPOUND', 'SUB_AREA', 'CITY', 'EMIRATE', 'COUNTRY']
    
    def __init__(self, csv_path: str, preserve_order: bool = True):
        """
        初始化分析器
        
        Args:
            csv_path: 标注数据CSV文件路径
            preserve_order: 是否保留CSV中的列顺序（True=排列，False=组合）
        """
        self.csv_path = Path(csv_path)
        self.preserve_order = preserve_order
        self.data = []
        self.templates = []
        self.template_counts = Counter()
        
    def load_data(self):
        """加载CSV数据"""
        print(f"📖 正在加载数据: {self.csv_path}")
        
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            self.data = list(reader)
        
        order_mode = "排列顺序（基于实体在地址中的实际位置）" if self.preserve_order else "组合关系（忽略顺序）"
        print(f"✅ 加载完成，共 {len(self.data)} 条记录")
        print(f"📋 分析模式: {order_mode}\n")
    
    def _has_value(self, value: str) -> bool:
        """
        判断字段是否有有效值
        
        Args:
            value: 字段值
            
        Returns:
            是否有有效值
        """
        if not value:
            return False
        value = value.strip()
        return bool(value) and value != ''
    
    def _find_entity_position(self, address: str, entity_value: str) -> int:
        """
        查找实体在地址中的位置
        
        Args:
            address: 完整地址
            entity_value: 实体值（可能包含多个用|分隔）
            
        Returns:
            实体在地址中的最小位置（找不到返回无穷大）
        """
        address_lower = address.lower()
        
        # 处理多值情况（用 | 分隔）
        values = [v.strip() for v in entity_value.split('|')]
        
        min_pos = float('inf')
        for value in values:
            if not value:
                continue
            value_lower = value.lower()
            pos = address_lower.find(value_lower)
            if pos != -1 and pos < min_pos:
                min_pos = pos
        
        return min_pos
    
    def _extract_template(self, row: Dict[str, str]) -> List[str]:
        """
        从一条记录中提取模板
        
        Args:
            row: CSV行数据
            
        Returns:
            实体类型列表（按实际位置排序或只看组合）
        """
        # 收集所有有值的实体
        entities_with_type = []
        
        for entity_type in self.ENTITY_TYPES:
            if entity_type in row and self._has_value(row[entity_type]):
                entities_with_type.append(entity_type)
        
        if not entities_with_type:
            return []
        
        if self.preserve_order and 'formatted_address' in row:
            # 按实体在地址中的实际位置排序
            address = row['formatted_address']
            entity_positions = []
            
            for entity_type in entities_with_type:
                entity_value = row[entity_type]
                position = self._find_entity_position(address, entity_value)
                entity_positions.append((position, entity_type))
            
            # 按位置排序
            entity_positions.sort(key=lambda x: x[0])
            template = [entity_type for _, entity_type in entity_positions]
        else:
            # 只关注组合，按预定义顺序
            template = entities_with_type
        
        return template
    
    def analyze(self):
        """分析所有数据，提取模板统计"""
        print("🔍 开始分析模板...")
        
        for row in self.data:
            template = self._extract_template(row)
            if template:  # 只统计至少有一个实体的记录
                template_tuple = tuple(template)
                self.templates.append(template_tuple)
                self.template_counts[template_tuple] += 1
        
        print(f"✅ 分析完成，发现 {len(self.template_counts)} 种不同的模板\n")
    
    def _format_template_pattern(self, template: Tuple[str]) -> str:
        """
        将模板转换为配置文件中的pattern格式
        
        Args:
            template: 实体类型元组
            
        Returns:
            格式化的pattern字符串
        """
        return ' '.join(f'{{{entity}}}' for entity in template)
    
    def print_statistics(self, top_n: int = 20):
        """
        打印统计结果
        
        Args:
            top_n: 显示前N个最常见的模板
        """
        print("="*80)
        print("📊 模板统计结果")
        print("="*80)
        
        total_records = len(self.templates)
        separator = ' → ' if self.preserve_order else ' + '
        mode_desc = "（保留排列顺序）" if self.preserve_order else "（仅组合关系）"
        
        print(f"\n总记录数: {total_records}")
        print(f"模板种类数: {len(self.template_counts)}")
        print(f"分析模式: {mode_desc}")
        print(f"\n前 {top_n} 个最常见的模板:\n")
        
        print(f"{'排名':<6} {'出现次数':<10} {'占比':<10} {'模板'}")
        print("-" * 80)
        
        for rank, (template, count) in enumerate(self.template_counts.most_common(top_n), 1):
            percentage = (count / total_records) * 100
            template_str = separator.join(template)
            print(f"{rank:<6} {count:<10} {percentage:>6.2f}%    {template_str}")
    
    def get_entity_statistics(self) -> Dict[str, int]:
        """
        统计各实体类型的出现频率
        
        Returns:
            实体类型 -> 出现次数的字典
        """
        entity_counts = defaultdict(int)
        
        for row in self.data:
            for entity_type in self.ENTITY_TYPES:
                if entity_type in row and self._has_value(row[entity_type]):
                    entity_counts[entity_type] += 1
        
        return dict(entity_counts)
    
    def print_entity_statistics(self):
        """打印实体统计"""
        print("\n" + "="*80)
        print("📈 实体出现频率统计")
        print("="*80)
        
        entity_counts = self.get_entity_statistics()
        total_records = len(self.data)
        
        print(f"\n{'实体类型':<15} {'出现次数':<12} {'出现率':<10} {'进度条'}")
        print("-" * 80)
        
        for entity_type in self.ENTITY_TYPES:
            count = entity_counts.get(entity_type, 0)
            percentage = (count / total_records) * 100 if total_records > 0 else 0
            bar_length = int(percentage / 2)  # 最大50个字符
            bar = '█' * bar_length
            
            print(f"{entity_type:<15} {count:<12} {percentage:>6.2f}%    {bar}")
    
    def generate_config(self, output_path: str = None, min_count: int = 2, 
                       max_templates: int = 15) -> Dict:
        """
        生成配置文件格式的模板定义
        
        Args:
            output_path: 输出文件路径（可选）
            min_count: 最小出现次数（过滤掉太少的模板）
            max_templates: 最大模板数量
            
        Returns:
            配置字典
        """
        print("\n" + "="*80)
        print("⚙️  生成配置文件格式")
        print("="*80)
        
        # 筛选符合条件的模板
        filtered_templates = [
            (template, count) 
            for template, count in self.template_counts.most_common()
            if count >= min_count
        ][:max_templates]
        
        total_count = sum(count for _, count in filtered_templates)
        
        # 生成配置
        config_templates = []
        for template, count in filtered_templates:
            weight = round(count / total_count, 3)
            pattern = self._format_template_pattern(template)
            
            config_templates.append({
                "pattern": pattern,
                "weight": weight,
                "comment": f"出现 {count} 次"
            })
        
        # 归一化权重（确保总和为1.0）
        total_weight = sum(t['weight'] for t in config_templates)
        if total_weight > 0:
            for t in config_templates:
                t['weight'] = round(t['weight'] / total_weight, 3)
        
        # 调整最后一个权重确保总和精确为1.0
        if config_templates:
            weight_sum = sum(t['weight'] for t in config_templates[:-1])
            config_templates[-1]['weight'] = round(1.0 - weight_sum, 3)
        
        config = {
            "templates": {
                "definitions": config_templates,
                "separator_variations": {
                    "options": [",", "-", "."],
                    "probability": 0.6
                }
            }
        }
        
        # 打印生成的配置
        print(f"\n生成了 {len(config_templates)} 个模板配置:\n")
        
        for i, template in enumerate(config_templates, 1):
            print(f"{i}. {template['pattern']}")
            print(f"   权重: {template['weight']:.3f} ({template['comment']})")
        
        # 保存到文件
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            
            print(f"\n✅ 配置已保存到: {output_file}")
        else:
            print("\n💡 提示: 使用 --output 参数可以将配置保存到文件")
        
        return config
    
    def analyze_template_combinations(self):
        """分析实体组合的规律"""
        print("\n" + "="*80)
        print("🔬 实体组合分析")
        print("="*80)
        
        # 分析每个长度的模板
        length_distribution = defaultdict(list)
        separator = ' → ' if self.preserve_order else ' + '
        
        for template, count in self.template_counts.items():
            length = len(template)
            length_distribution[length].append((template, count))
        
        print("\n按实体数量分组:\n")
        
        for length in sorted(length_distribution.keys()):
            templates = length_distribution[length]
            total = sum(count for _, count in templates)
            
            print(f"【{length} 个实体】共 {len(templates)} 种模板，出现 {total} 次")
            
            # 显示该长度下的前5个最常见模板
            top_templates = sorted(templates, key=lambda x: x[1], reverse=True)[:5]
            for template, count in top_templates:
                template_str = separator.join(template)
                print(f"  • {template_str:<50} ({count} 次)")
            print()
    
    def get_recommendations(self):
        """提供模板配置建议"""
        print("\n" + "="*80)
        print("💡 配置建议")
        print("="*80)
        
        total_records = len(self.templates)
        top_10_count = sum(count for _, count in self.template_counts.most_common(10))
        coverage = (top_10_count / total_records) * 100
        
        print("\n1. 数据覆盖度分析:")
        print(f"   • 前10个模板覆盖了 {top_10_count}/{total_records} 条记录 ({coverage:.1f}%)")
        
        if coverage >= 80:
            print("   ✅ 覆盖度很好！使用前10-15个模板即可")
        elif coverage >= 60:
            print("   ⚠️  覆盖度中等，建议使用前15-20个模板")
        else:
            print("   ⚠️  数据较分散，建议增加更多模板或检查数据质量")
        
        # 分析必需实体
        entity_counts = self.get_entity_statistics()
        total = len(self.data)
        
        print("\n2. 必需实体识别（出现率>80%）:")
        essential_entities = []
        for entity_type in self.ENTITY_TYPES:
            rate = (entity_counts.get(entity_type, 0) / total) * 100
            if rate > 80:
                essential_entities.append(entity_type)
                print(f"   • {entity_type}: {rate:.1f}%")
        
        if essential_entities:
            print("\n   💡 建议: 确保大部分模板包含这些实体")
        
        # 分析常见组合
        print("\n3. 推荐的模板配置策略:")
        print(f"   • 使用前10个最常见模板（占比约{coverage:.0f}%）")
        print("   • 为每个模板分配基于出现频率的权重")
        print("   • 保留一些低频模板以增加多样性")
        
        print("\n4. 生成数据建议:")
        print("   • generate_size: 建议设置为真实数据的2-5倍")
        print(f"   • 真实数据量: {total} 条")
        print(f"   • 建议生成: {total * 3} - {total * 5} 条")


def main():
    """主函数"""
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(
        description='分析标注数据，生成地址模板配置',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 分析数据并显示统计
  python3 %(prog)s data/ner/raw_data/第三次训练/线上数据-第三次训练-01-标注结果.csv
  
  # 生成配置文件
  python3 %(prog)s data/ner/raw_data/第三次训练/线上数据-第三次训练-01-标注结果.csv \\
      --output data/ner/simulator/uae/config/generated_templates.json
  
  # 分析并保存前30个最常见的模板
  python3 %(prog)s <csv_file> --top 30 -o templates.json
  
  # 只保存出现≥5次的模板
  python3 %(prog)s <csv_file> --min-count 5 -o templates.json
        """
    )
    
    parser.add_argument('csv_file', help='标注数据CSV文件路径')
    parser.add_argument('--output', '-o', help='输出配置文件路径')
    parser.add_argument('--top', '-t', type=int, default=20, 
                       help='显示和保存的最大模板数量 (默认: 20)')
    parser.add_argument('--min-count', type=int, default=1,
                       help='最小出现次数，低于此次数的模板将被过滤 (默认: 1)')
    parser.add_argument('--no-order', action='store_true',
                       help='不保留排列顺序，只分析组合关系')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not Path(args.csv_file).exists():
        print(f"❌ 错误: 文件不存在: {args.csv_file}")
        sys.exit(1)
    
    try:
        # 创建分析器（默认保留排列顺序）
        preserve_order = not args.no_order
        analyzer = TemplateAnalyzer(args.csv_file, preserve_order=preserve_order)
        
        # 加载数据
        analyzer.load_data()
        
        # 分析
        analyzer.analyze()
        
        # 显示统计结果
        analyzer.print_statistics(top_n=args.top)
        analyzer.print_entity_statistics()
        analyzer.analyze_template_combinations()
        
        # 生成配置
        analyzer.generate_config(
            output_path=args.output,
            min_count=args.min_count,
            max_templates=args.top
        )
        
        # 提供建议
        analyzer.get_recommendations()
        
        print("\n" + "="*80)
        print("✅ 分析完成！")
        print("="*80)
        
    except (FileNotFoundError, ValueError, KeyError, UnicodeDecodeError) as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:  # pylint: disable=broad-except
        # 捕获其他未预期的错误
        print(f"\n❌ 未预期的错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

