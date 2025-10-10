#!/usr/bin/env python3
"""从标注数据中提取实体到词典文件"""

import csv
import os
from collections import defaultdict
from pathlib import Path

def extract_entities_from_csv(csv_paths, output_dir):
    """从CSV文件中提取实体"""
    entities = defaultdict(set)
    
    entity_types = ['BUILDING', 'STREET', 'COMPOUND', 'SUB_AREA', 'CITY', 'EMIRATE', 'COUNTRY']
    
    for csv_path in csv_paths:
        print(f"Processing {csv_path}...")
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                for entity_type in entity_types:
                    value = row.get(entity_type, '').strip()
                    if value:
                        # 处理多值情况（用 | 分隔）
                        values = [v.strip() for v in value.split('|')]
                        for v in values:
                            if v:
                                entities[entity_type].add(v)
    
    # 写入词典文件
    os.makedirs(output_dir, exist_ok=True)
    
    for entity_type in entity_types:
        output_file = os.path.join(output_dir, f"{entity_type.lower()}.txt")
        sorted_entities = sorted(entities[entity_type])
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for entity in sorted_entities:
                f.write(entity + '\n')
        
        print(f"✓ {entity_type}: {len(sorted_entities)} unique entities -> {output_file}")
    
    return entities

if __name__ == "__main__":
    # 标注数据文件路径
    csv_files = [
        "data/ner/raw_data/第三次训练/线上数据-第三次训练-标注结果.csv"
    ]
    
    # 输出目录
    output_dir = "data/ner/simulator/uae/dictionaries"
    
    # 提取实体
    entities = extract_entities_from_csv(csv_files, output_dir)
    
    print("\n" + "="*50)
    print("实体提取完成！")
    print("="*50)

