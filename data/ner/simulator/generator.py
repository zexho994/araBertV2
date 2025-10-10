#!/usr/bin/env python3
"""
地址数据模拟器 - 主生成器
根据配置文件和实体词典生成合成地址数据

生成流程:
T1: 从模板和词典生成原始地址
T2: 为实体间添加分隔符
T3: 应用大小写变体
T4: 注入噪音（标点、数字、无意义词）
"""

import json
import csv
import random
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple


class AddressGenerator:
    """地址生成器核心类"""
    
    def __init__(self, config_path: str):
        """
        初始化生成器
        
        Args:
            config_path: 配置文件路径（JSON5格式）
        """
        self.config_path = Path(config_path).resolve()  # 转换为绝对路径
        self.config = self._load_config()
        self.dictionaries = self._load_dictionaries()
        self.generated_addresses = set()  # 用于去重
        
    def _load_config(self) -> Dict:
        """
        加载配置文件
        支持JSON5格式（移除注释和尾随逗号）
        
        Returns:
            配置字典
        """
        print(f"📖 正在加载配置文件: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 简单处理JSON5：移除单行注释
        lines = []
        for line in content.split('\n'):
            # 移除 // 开头的注释
            if '//' not in line:
                lines.append(line)
            else:
                # 保留 // 之前的内容
                lines.append(line.split('//')[0])
        
        content = '\n'.join(lines)
        
        # 移除尾随逗号（JSON5特性）
        content = re.sub(r',(\s*[}\]])', r'\1', content)
        
        config = json.loads(content)
        print("✅ 配置加载成功")
        return config
    
    def _load_dictionaries(self) -> Dict[str, List[str]]:
        """
        加载所有实体词典
        
        Returns:
            实体类型 -> 实体列表的字典
        """
        print("\n📚 正在加载实体词典...")
        dictionaries = {}
        
        # 获取项目根目录
        # 配置文件在 data/ner/simulator/uae/config/generator_config.json
        # 向上6级到项目根目录 (config -> uae -> simulator -> ner -> data -> araBertv2)
        project_root = self.config_path.parent.parent.parent.parent.parent.parent
        
        for entity_type, dict_path in self.config['dictionaries'].items():
            # 相对于项目根目录解析路径
            full_path = project_root / dict_path
            
            with open(full_path, 'r', encoding='utf-8') as f:
                # 读取所有非空行
                entities = [line.strip() for line in f if line.strip()]
                dictionaries[entity_type] = entities
                
            print(f"  ✓ {entity_type}: {len(entities)} 个实体")
        
        print("✅ 词典加载完成\n")
        return dictionaries
    
    def _select_template(self) -> Tuple[str, List[str]]:
        """
        根据权重随机选择一个地址模板
        
        Returns:
            (模板字符串, 模板中的实体类型列表)
        """
        templates = self.config['templates']['definitions']
        
        # 提取权重
        weights = [t['weight'] for t in templates]
        
        # 根据权重随机选择
        selected = random.choices(templates, weights=weights)[0]
        pattern = selected['pattern']
        
        # 提取模板中的实体类型
        entity_types = re.findall(r'\{(\w+)\}', pattern)
        
        return pattern, entity_types
    
    def _generate_t1(self) -> Tuple[str, Dict[str, str], List[str]]:
        """
        T1: 从模板和词典生成原始地址
        
        Returns:
            (地址字符串, 实体字典, 实体值列表（按模板顺序）)
        """
        # 选择模板
        pattern, entity_types = self._select_template()
        
        # 为每个实体类型随机选择一个值
        entities = {}
        entity_values_ordered = []  # 保存实体值的顺序列表
        
        for entity_type in entity_types:
            # 从对应词典中随机选择
            entity_value = random.choice(self.dictionaries[entity_type])
            entities[entity_type] = entity_value
            entity_values_ordered.append(entity_value)
        
        # 填充模板（此时实体之间用空格分隔）
        address = pattern
        for entity_type, entity_value in entities.items():
            address = address.replace(f'{{{entity_type}}}', entity_value)
        
        return address, entities, entity_values_ordered
    
    def _generate_t2(self, address: str, entity_values: List[str]) -> str:
        """
        T2: 为实体间添加分隔符
        注意：只在实体之间添加分隔符，不破坏实体内部的空格
        
        Args:
            address: T1生成的地址
            entity_values: 实体值列表（按在地址中出现的顺序）
            
        Returns:
            添加分隔符后的地址
        """
        separator_config = self.config['templates']['separator_variations']
        probability = separator_config['probability']
        separator_options = separator_config['options']
        
        # 按概率决定是否添加分隔符
        if random.random() > probability:
            return address
        
        # 随机选择一个分隔符
        separator = random.choice(separator_options)
        
        # 在实体之间添加分隔符
        # 策略：找到每个实体在地址中的位置，在实体之间插入分隔符
        result = address
        offset = 0  # 由于插入分隔符导致的位置偏移
        
        for i in range(len(entity_values) - 1):
            current_entity = entity_values[i]
            next_entity = entity_values[i + 1]
            
            # 找到当前实体的结束位置
            current_pos = result.find(current_entity, offset)
            if current_pos == -1:
                continue
            
            current_end = current_pos + len(current_entity)
            
            # 找到下一个实体的开始位置
            next_pos = result.find(next_entity, current_end)
            if next_pos == -1:
                continue
            
            # 在两个实体之间插入分隔符
            # 提取实体之间的间隔部分（通常是空格）
            between = result[current_end:next_pos]
            
            # 如果实体之间有空格，在空格后添加分隔符
            if between.strip() == '':
                # 替换为: 空格 + 分隔符 + 空格
                new_between = f' {separator} '
                result = result[:current_end] + new_between + result[next_pos:]
                offset = next_pos + len(new_between) - len(between)
            else:
                offset = next_pos
        
        return result
    
    def _apply_case_variation(self, text: str) -> str:
        """
        应用大小写变体
        
        Args:
            text: 原始文本
            
        Returns:
            变体后的文本
        """
        case_config = self.config['typo_injection']['case_variations']
        
        if not case_config['enabled']:
            return text
        
        # 按概率决定是否应用变体
        if random.random() > case_config['probability']:
            return text
        
        # 根据策略权重选择变体方式
        strategies = case_config['strategies']
        strategy = random.choices(
            list(strategies.keys()),
            weights=list(strategies.values())
        )[0]
        
        if strategy == 'normal_case':
            return text
        elif strategy == 'all_lowercase':
            return text.lower()
        elif strategy == 'all_uppercase':
            return text.upper()
        elif strategy == 'random_case':
            # 随机大小写每个字符
            return ''.join(
                c.upper() if random.random() > 0.5 else c.lower()
                for c in text
            )
        
        return text
    
    def _generate_t3(self, address: str, entities: Dict[str, str]) -> Tuple[str, Dict[str, str]]:
        """
        T3: 应用大小写变体
        注意：这里对整个地址应用变体，但保持实体值不变（用于输出）
        
        Args:
            address: T2生成的地址
            entities: 原始实体字典
            
        Returns:
            (变体后的地址, 更新后的实体字典)
        """
        typo_config = self.config['typo_injection']
        
        if not typo_config['enabled']:
            return address, entities
        
        # 按全局概率决定是否进行typo注入
        if random.random() > typo_config['global_probability']:
            return address, entities
        
        # 对整个地址应用大小写变体
        address_t3 = self._apply_case_variation(address)
        
        # 同时更新实体值（保持一致性）
        entities_t3 = {}
        for entity_type, entity_value in entities.items():
            # 为每个实体独立应用变体
            entities_t3[entity_type] = self._apply_case_variation(entity_value)
        
        return address_t3, entities_t3
    
    def _generate_noise_punctuation(self) -> str:
        """生成标点噪音"""
        noise_config = self.config['noise_injection']['noise_types']['punctuation']
        chars = noise_config['characters']
        max_consecutive = noise_config['max_consecutive']
        
        # 随机选择1到max_consecutive个标点
        count = random.randint(1, max_consecutive)
        return ''.join(random.choices(chars, k=count))
    
    def _generate_noise_number(self) -> str:
        """生成数字噪音"""
        patterns = self.config['noise_injection']['noise_types']['numbers']['patterns']
        
        # 根据概率选择数字模式
        pattern_types = list(patterns.keys())
        pattern_weights = [patterns[p]['probability'] for p in pattern_types]
        selected_pattern = random.choices(pattern_types, weights=pattern_weights)[0]
        
        if selected_pattern == 'random_digits':
            min_len = patterns['random_digits']['min_length']
            max_len = patterns['random_digits']['max_length']
            length = random.randint(min_len, max_len)
            return ''.join(str(random.randint(0, 9)) for _ in range(length))
        
        elif selected_pattern == 'phone_like':
            # 生成类似电话号码的格式
            # +971XXXXXXXXX -> +971 + 9位随机数字
            return '+971' + ''.join(str(random.randint(0, 9)) for _ in range(9))
        
        elif selected_pattern == 'po_box':
            # 生成P.O. Box格式
            box_number = ''.join(str(random.randint(0, 9)) for _ in range(5))
            return f'P.O. Box {box_number}'
        
        return ''
    
    def _generate_noise_word(self) -> str:
        """生成无意义词噪音"""
        word_config = self.config['noise_injection']['noise_types']['meaningless_words']
        return random.choice(word_config['word_list'])
    
    def _generate_t4(self, address: str) -> str:
        """
        T4: 注入噪音（标点、数字、无意义词）
        注意：噪音注入在实体之间，不破坏实体本身
        
        Args:
            address: T3生成的地址
            
        Returns:
            注入噪音后的地址
        """
        noise_config = self.config['noise_injection']
        
        if not noise_config['enabled']:
            return address
        
        # 按全局概率决定是否注入噪音
        if random.random() > noise_config['global_noise_probability']:
            return address
        
        # 将地址按分隔符和空格分割成部分
        # 使用正则保留分隔符
        parts = re.split(r'(\s+|,\s*|/\s*|-\s*)', address)
        
        # 确定注入噪音的数量
        max_noise = noise_config['max_noise_per_address']
        noise_count = random.randint(1, max_noise)
        
        # 收集可以注入噪音的位置（分隔符位置）
        separator_indices = [i for i, part in enumerate(parts) 
                           if re.match(r'^\s+|,\s*|/\s*|-\s*$', part)]
        
        if not separator_indices:
            return address
        
        # 随机选择注入位置
        inject_positions = random.sample(
            separator_indices,
            min(noise_count, len(separator_indices))
        )
        
        # 在选定位置注入噪音
        noise_types = noise_config['noise_types']
        
        for pos in inject_positions:
            # 根据概率选择噪音类型
            noise_options = []
            noise_weights = []
            
            if noise_types['punctuation']['enabled']:
                noise_options.append('punctuation')
                noise_weights.append(noise_types['punctuation']['probability'])
            
            if noise_types['numbers']['enabled']:
                noise_options.append('numbers')
                noise_weights.append(noise_types['numbers']['probability'])
            
            if noise_types['meaningless_words']['enabled']:
                noise_options.append('meaningless_words')
                noise_weights.append(noise_types['meaningless_words']['probability'])
            
            if not noise_options:
                continue
            
            # 选择噪音类型
            noise_type = random.choices(noise_options, weights=noise_weights)[0]
            
            # 生成噪音
            if noise_type == 'punctuation':
                noise = self._generate_noise_punctuation()
            elif noise_type == 'numbers':
                noise = self._generate_noise_number()
            elif noise_type == 'meaningless_words':
                noise = self._generate_noise_word()
            else:
                continue
            
            # 在分隔符位置添加噪音
            # 格式: "原分隔符 噪音 "
            parts[pos] = parts[pos] + noise + ' '
        
        return ''.join(parts).strip()
    
    def generate_one_address(self) -> Tuple[str, Dict[str, str]]:
        """
        生成一个完整的地址
        执行 T1 -> T2 -> T3 -> T4 的完整流程
        
        Returns:
            (最终地址字符串, 实体字典)
        """
        # T1: 生成原始地址
        address_t1, entities, entity_values_ordered = self._generate_t1()
        
        # T2: 添加分隔符（传入实体值列表，确保只在实体之间添加）
        address_t2 = self._generate_t2(address_t1, entity_values_ordered)
        
        # T3: 应用大小写变体
        address_t3, entities_t3 = self._generate_t3(address_t2, entities)
        
        # T4: 注入噪音
        address_t4 = self._generate_t4(address_t3)
        
        return address_t4, entities_t3
    
    def _validate_address(self, address: str) -> bool:
        """
        验证生成的地址是否符合质量要求
        
        Args:
            address: 待验证的地址
            
        Returns:
            是否通过验证
        """
        validation_config = self.config['output']['quality_validation']
        
        if not validation_config['enabled']:
            return True
        
        # 检查长度
        length = len(address)
        min_length = validation_config['min_address_length']
        max_length = validation_config['max_address_length']
        
        if length < min_length or length > max_length:
            return False
        
        return True
    
    def generate_addresses(self) -> List[Dict[str, str]]:
        """
        生成指定数量的地址
        
        Returns:
            地址记录列表，每条记录包含formatted_address和各实体字段
        """
        target_size = self.config['generate_size']
        unique_only = self.config.get('unique_combinations_only', True)
        
        print("🚀 开始生成地址...")
        print(f"   目标数量: {target_size}")
        print(f"   唯一性要求: {'是' if unique_only else '否'}\n")
        
        results = []
        attempts = 0
        max_attempts = target_size * 10  # 最多尝试10倍数量
        
        while len(results) < target_size and attempts < max_attempts:
            attempts += 1
            
            # 生成一个地址
            address, entities = self.generate_one_address()
            
            # 验证地址质量
            if not self._validate_address(address):
                continue
            
            # 检查唯一性
            if unique_only and address in self.generated_addresses:
                continue
            
            # 添加到结果
            self.generated_addresses.add(address)
            
            # 构建输出记录
            record = {'formatted_address': address}
            
            # 添加各实体字段（按配置中的顺序）
            for field in self.config['output']['fields']:
                if field == 'formatted_address':
                    continue
                record[field] = entities.get(field, '')
            
            results.append(record)
            
            # 显示进度
            if len(results) % 10 == 0:
                print(f"   已生成: {len(results)}/{target_size}")
        
        if len(results) < target_size:
            print(f"\n⚠️  警告: 只生成了 {len(results)} 条地址（目标 {target_size}）")
            print(f"   尝试次数: {attempts}")
        else:
            print(f"\n✅ 成功生成 {len(results)} 条地址")
        
        return results
    
    def save_to_csv(self, records: List[Dict[str, str]]) -> str:
        """
        将生成的地址保存到CSV文件
        
        Args:
            records: 地址记录列表
            
        Returns:
            输出文件路径
        """
        output_config = self.config['output']
        
        # 构建输出文件名
        output_dir = self.config_path.parent / output_config['output_dir']
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = output_config['file_prefix']
        if output_config.get('timestamp_suffix', True):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename += timestamp
        filename += '.csv'
        
        output_path = output_dir / filename
        
        # 是否需要打乱顺序
        if output_config.get('shuffle', False):
            random.shuffle(records)
        
        # 写入CSV
        print(f"\n💾 正在保存到文件: {output_path}")
        
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            if records:
                fieldnames = output_config['fields']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(records)
        
        print(f"✅ 保存成功！共 {len(records)} 条记录")
        
        return str(output_path)
    
    def run(self):
        """运行完整的生成流程"""
        print("="*60)
        print("🏠 地址数据模拟器")
        print("="*60 + "\n")
        
        # 生成地址
        records = self.generate_addresses()
        
        # 保存到文件
        if records:
            output_path = self.save_to_csv(records)
            
            # 显示统计信息
            print("\n" + "="*60)
            print("📊 生成统计")
            print("="*60)
            print(f"总记录数: {len(records)}")
            print(f"输出文件: {output_path}")
            print(f"平均地址长度: {sum(len(r['formatted_address']) for r in records) / len(records):.1f} 字符")
            print("="*60)


def main():
    """主函数"""
    import sys
    
    # 默认配置文件路径
    default_config = "data/ner/simulator/uae/config/generator_config.json"
    
    # 从命令行参数获取配置文件路径
    config_path = sys.argv[1] if len(sys.argv) > 1 else default_config
    
    try:
        # 创建生成器并运行
        generator = AddressGenerator(config_path)
        generator.run()
        
    except (FileNotFoundError, json.JSONDecodeError, ValueError, KeyError) as e:
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

