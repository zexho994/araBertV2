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
import string
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
    
    def _generate_t1(self) -> Tuple[str, Dict[str, str], List[str], List[str]]:
        """
        T1: 从模板和词典生成原始地址
        
        Returns:
            (地址字符串, 实体字典, 实体值列表（按模板顺序）, 实体类型列表（按模板顺序）)
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
        
        return address, entities, entity_values_ordered, entity_types
    
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
    
    def _apply_typo_to_entity(self, text: str, entity_type: str) -> str:
        """
        为单个实体应用拼写错误
        
        Args:
            text: 实体文本
            entity_type: 实体类型（用于查找配置）
            
        Returns:
            带有拼写错误的文本
        """
        # 检查是否有该实体类型的配置
        per_entity_config = self.config['typo_injection'].get('per_entity_config', {})
        if entity_type not in per_entity_config:
            return text
        
        entity_config = per_entity_config[entity_type]
        
        # 按概率决定是否为该实体注入错误
        if random.random() > entity_config['probability']:
            return text
        
        # 如果文本太短，不注入错误
        if len(text) < 2:
            return text
        
        # 计算最大错误数量
        max_errors = max(1, int(len(text) * entity_config['max_char_error_ratio']))
        error_count = random.randint(1, max_errors)
        
        # 转换为字符列表以便修改
        chars = list(text)
        typo_types = entity_config['typo_types']
        
        for _ in range(error_count):
            if len(chars) < 2:
                break
            
            # 根据权重选择错误类型
            typo_type = random.choices(
                list(typo_types.keys()),
                weights=list(typo_types.values())
            )[0]
            
            # 随机选择一个位置（避免空格）
            non_space_indices = [i for i, c in enumerate(chars) if c != ' ']
            if not non_space_indices:
                break
            
            pos = random.choice(non_space_indices)
            
            if typo_type == 'swap' and pos < len(chars) - 1:
                # 交换相邻字符（确保下一个不是空格）
                next_pos = pos + 1
                while next_pos < len(chars) and chars[next_pos] == ' ':
                    next_pos += 1
                if next_pos < len(chars):
                    chars[pos], chars[next_pos] = chars[next_pos], chars[pos]
            
            elif typo_type == 'deletion':
                # 删除字符
                chars.pop(pos)
            
            elif typo_type == 'insertion':
                # 插入随机字符（字母或数字）
                # 优先插入相似的字符（如果是字母插入字母，如果是数字插入数字）
                original_char = chars[pos]
                if original_char.isalpha():
                    # 插入随机字母，保持相同大小写风格
                    if original_char.isupper():
                        random_char = random.choice(string.ascii_uppercase)
                    else:
                        random_char = random.choice(string.ascii_lowercase)
                elif original_char.isdigit():
                    random_char = str(random.randint(0, 9))
                else:
                    random_char = random.choice('abcdefghijklmnopqrstuvwxyz')
                
                chars.insert(pos, random_char)
            
            elif typo_type == 'substitution':
                # 替换为随机字符
                original_char = chars[pos]
                if original_char.isalpha():
                    if original_char.isupper():
                        chars[pos] = random.choice(string.ascii_uppercase)
                    else:
                        chars[pos] = random.choice(string.ascii_lowercase)
                elif original_char.isdigit():
                    chars[pos] = str(random.randint(0, 9))
        
        return ''.join(chars)
    
    def _generate_t3(self, address: str, entities: Dict[str, str], entity_types: List[str]) -> Tuple[str, Dict[str, str], List[str]]:
        """
        T3: 应用拼写错误和大小写变体
        包括：
        1. 为每个实体应用拼写错误（swap, deletion, insertion, substitution）
        2. 应用大小写变体
        
        Args:
            address: T2生成的地址
            entities: 原始实体字典
            entity_types: 实体类型列表（按模板顺序）
            
        Returns:
            (变体后的地址, 更新后的实体字典, 变体后的实体值列表（按模板顺序）)
        """
        typo_config = self.config['typo_injection']
        
        if not typo_config['enabled']:
            # 按原顺序提取实体值
            entity_values_ordered = [entities[et] for et in entity_types if et in entities]
            return address, entities, entity_values_ordered
        
        # 按全局概率决定是否进行typo注入
        if random.random() > typo_config['global_probability']:
            entity_values_ordered = [entities[et] for et in entity_types if et in entities]
            return address, entities, entity_values_ordered
        
        # 步骤1: 为每个实体应用拼写错误
        entities_with_typos = {}
        address_with_typos = address
        
        for entity_type, entity_value in entities.items():
            # 应用拼写错误
            entity_with_typo = self._apply_typo_to_entity(entity_value, entity_type)
            entities_with_typos[entity_type] = entity_with_typo
            
            # 在地址中替换实体值
            # 注意：这里需要小心处理，避免误替换
            if entity_value in address_with_typos:
                address_with_typos = address_with_typos.replace(entity_value, entity_with_typo, 1)
        
        # 步骤2: 应用大小写变体（对整个地址）
        address_t3 = self._apply_case_variation(address_with_typos)
        
        # 同时更新实体值（应用大小写变体）
        entities_t3 = {}
        for entity_type, entity_value in entities_with_typos.items():
            entities_t3[entity_type] = self._apply_case_variation(entity_value)
        
        # 按原顺序提取T3后的实体值
        entity_values_ordered_t3 = [entities_t3[et] for et in entity_types if et in entities_t3]
        
        return address_t3, entities_t3, entity_values_ordered_t3
    
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
    
    def _generate_t4(self, address: str, entity_values: List[str]) -> str:
        """
        T4: 注入噪音（标点、数字、无意义词）
        注意：噪音只在实体之间注入，不破坏实体本身的完整性
        
        Args:
            address: T3生成的地址
            entity_values: T3后的实体值列表（已应用拼写错误和大小写变体）
            
        Returns:
            注入噪音后的地址
        """
        noise_config = self.config['noise_injection']
        
        if not noise_config['enabled']:
            return address
        
        # 按全局概率决定是否注入噪音
        if random.random() > noise_config['global_noise_probability']:
            return address
        
        # 如果没有实体，不注入噪音
        if not entity_values or len(entity_values) < 2:
            return address
        
        # 确定注入噪音的数量
        max_noise = noise_config['max_noise_per_address']
        noise_count = random.randint(1, max_noise)
        
        # 找到所有实体之间的间隔位置
        # 策略：找到每个实体的结束位置，在实体之间插入噪音
        injection_points = []  # 存储 (position, gap_text) 元组
        offset = 0
        
        for i in range(len(entity_values) - 1):
            current_entity = entity_values[i]
            next_entity = entity_values[i + 1]
            
            # 找到当前实体的位置
            current_pos = address.find(current_entity, offset)
            if current_pos == -1:
                continue
            
            current_end = current_pos + len(current_entity)
            
            # 找到下一个实体的位置
            next_pos = address.find(next_entity, current_end)
            if next_pos == -1:
                continue
            
            # 记录实体之间的间隔位置和内容
            gap_text = address[current_end:next_pos]
            if gap_text.strip():  # 只在有间隔的地方注入
                injection_points.append((current_end, next_pos, gap_text))
            
            offset = next_pos
        
        # 如果没有可注入的位置，返回原地址
        if not injection_points:
            return address
        
        # 随机选择注入位置
        selected_points = random.sample(
            injection_points,
            min(noise_count, len(injection_points))
        )
        
        # 按位置倒序排序，从后往前注入（避免位置偏移问题）
        selected_points.sort(key=lambda x: x[0], reverse=True)
        
        # 在选定位置注入噪音
        noise_types = noise_config['noise_types']
        result = address
        
        for start_pos, end_pos, gap_text in selected_points:
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
            
            # 在间隔中注入噪音
            # 保留原有的间隔符号，在其后添加噪音
            new_gap = gap_text.rstrip() + ' ' + noise + ' '
            result = result[:start_pos] + new_gap + result[end_pos:]
        
        return result.strip()
    
    def generate_one_address(self) -> Tuple[str, Dict[str, str]]:
        """
        生成一个完整的地址
        执行 T1 -> T2 -> T3 -> T4 的完整流程
        
        Returns:
            (最终地址字符串, 实体字典)
        """
        # T1: 生成原始地址
        address_t1, entities, entity_values_ordered, entity_types = self._generate_t1()
        
        # T2: 添加分隔符（传入实体值列表，确保只在实体之间添加）
        address_t2 = self._generate_t2(address_t1, entity_values_ordered)
        
        # T3: 应用拼写错误和大小写变体
        address_t3, entities_t3, entity_values_ordered_t3 = self._generate_t3(address_t2, entities, entity_types)
        
        # T4: 注入噪音（传入T3后的实体值列表，确保不破坏实体）
        address_t4 = self._generate_t4(address_t3, entity_values_ordered_t3)
        
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

