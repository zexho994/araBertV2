#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
词典去重工具

用于对词典文件进行去重处理，去重时忽略大小写。
支持处理任何国家/地区的词典文件。

使用方法：
    python3 data/ner/simulator/dedup_dictionaries.py -d uae/dictionaries
    python3 data/ner/simulator/dedup_dictionaries.py -d path/to/dictionaries
"""

from pathlib import Path
from typing import List, Set, Dict
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DictionaryDeduplicator:
    """词典去重工具类"""
    
    def __init__(self, directory: str = None):
        """
        初始化去重工具
        
        Args:
            directory: 词典文件所在目录，支持相对路径（相对于simulator/）或绝对路径
        """
        if directory is None:
            # 如果未指定目录，使用脚本所在目录
            self.directory = Path(__file__).parent
        else:
            dir_path = Path(directory)
            
            # 如果是相对路径，相对于simulator目录解析
            if not dir_path.is_absolute():
                # 脚本在 simulator/ 目录下
                simulator_dir = Path(__file__).parent
                self.directory = (simulator_dir / directory).resolve()
            else:
                self.directory = dir_path
        
        if not self.directory.exists():
            raise ValueError(f"目录不存在: {self.directory}")
    
    def _read_file(self, file_path: Path) -> List[str]:
        """
        读取文件内容
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件行列表
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.readlines()
        except (IOError, OSError) as e:
            logger.error("读取文件失败 %s: %s", file_path, e)
            return []
    
    def _write_file(self, file_path: Path, lines: List[str]) -> bool:
        """
        写入文件内容
        
        Args:
            file_path: 文件路径
            lines: 要写入的行列表
            
        Returns:
            是否写入成功
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            return True
        except (IOError, OSError) as e:
            logger.error("写入文件失败 %s: %s", file_path, e)
            return False
    
    def deduplicate_lines(self, lines: List[str]) -> tuple[List[str], Dict[str, int]]:
        """
        对行列表进行去重，忽略大小写
        保留每个词条第一次出现时的原始大小写形式
        
        Args:
            lines: 原始行列表
            
        Returns:
            (去重后的行列表, 统计信息字典)
        """
        seen_lower: Set[str] = set()  # 用于存储已见过的小写形式
        unique_lines: List[str] = []
        duplicate_count = 0
        empty_count = 0
        
        for line in lines:
            # 去除首尾空白字符
            stripped_line = line.strip()
            
            # 跳过空行
            if not stripped_line:
                empty_count += 1
                continue
            
            # 转换为小写用于比较
            lower_line = stripped_line.lower()
            
            # 如果是新词条，则保留（保留原始大小写）
            if lower_line not in seen_lower:
                seen_lower.add(lower_line)
                # 保留原始行（包括换行符）
                unique_lines.append(line if line.endswith('\n') else line + '\n')
            else:
                duplicate_count += 1
        
        stats = {
            'original_count': len(lines),
            'unique_count': len(unique_lines),
            'duplicate_count': duplicate_count,
            'empty_count': empty_count,
            'removed_count': duplicate_count + empty_count
        }
        
        return unique_lines, stats
    
    def process_file(self, file_path: Path, dry_run: bool = False) -> bool:
        """
        处理单个文件
        
        Args:
            file_path: 文件路径
            dry_run: 是否为演练模式（不实际写入文件）
            
        Returns:
            是否处理成功
        """
        logger.info("处理文件: %s", file_path.name)
        
        # 读取文件
        lines = self._read_file(file_path)
        if not lines:
            logger.warning("文件为空或读取失败: %s", file_path.name)
            return False
        
        # 去重
        unique_lines, stats = self.deduplicate_lines(lines)
        
        # 输出统计信息
        logger.info(
            "  原始行数: %d, 去重后: %d, 重复: %d, 空行: %d",
            stats['original_count'],
            stats['unique_count'],
            stats['duplicate_count'],
            stats['empty_count']
        )
        
        # 如果有变化且不是演练模式，则写入文件
        if stats['removed_count'] > 0:
            if not dry_run:
                success = self._write_file(file_path, unique_lines)
                if success:
                    logger.info("  ✓ 已更新文件")
                return success
            else:
                logger.info("  → 演练模式，未写入文件")
        else:
            logger.info("  ✓ 文件无需更新")
        
        return True
    
    def process_directory(self, pattern: str = "*.txt", dry_run: bool = False) -> Dict[str, bool]:
        """
        处理目录下所有匹配的文件
        
        Args:
            pattern: 文件匹配模式，默认为 "*.txt"
            dry_run: 是否为演练模式（不实际写入文件）
            
        Returns:
            处理结果字典 {文件名: 是否成功}
        """
        logger.info("开始处理目录: %s", self.directory)
        logger.info("文件模式: %s", pattern)
        logger.info("演练模式: %s", '是' if dry_run else '否')
        logger.info("-" * 60)
        
        # 查找所有匹配的文件
        files = list(self.directory.glob(pattern))
        
        if not files:
            logger.warning("未找到匹配的文件: %s", pattern)
            return {}
        
        logger.info("找到 %d 个文件\n", len(files))
        
        # 处理每个文件
        results = {}
        for file_path in sorted(files):
            # 跳过Python脚本文件
            if file_path.suffix == '.py':
                continue
            
            success = self.process_file(file_path, dry_run=dry_run)
            results[file_path.name] = success
            print()  # 空行分隔
        
        # 输出总结
        logger.info("-" * 60)
        logger.info("处理完成!")
        logger.info("总文件数: %d", len(results))
        logger.info("成功: %d", sum(results.values()))
        logger.info("失败: %d", len(results) - sum(results.values()))
        
        return results


def main():
    """主函数 - 命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='词典文件去重工具 - 支持任何国家/地区的词典文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 去重UAE词典（相对于simulator目录）
  python3 data/ner/simulator/dedup_dictionaries.py -d uae/dictionaries
  
  # 演练模式（不实际修改文件）
  python3 data/ner/simulator/dedup_dictionaries.py -d uae/dictionaries --dry-run
  
  # 使用绝对路径
  python3 data/ner/simulator/dedup_dictionaries.py \\
      -d /Users/zexho/Documents/python_script/araBertv2/data/ner/simulator/uae/dictionaries
  
  # 处理特定模式的文件
  python3 data/ner/simulator/dedup_dictionaries.py -d uae/dictionaries -p "city*.txt"
  
  # 如果在simulator目录下执行
  cd data/ner/simulator
  python3 dedup_dictionaries.py -d uae/dictionaries
        """
    )
    
    parser.add_argument(
        '-d', '--directory',
        type=str,
        required=True,
        help='词典文件所在目录（相对于simulator/目录或绝对路径）'
    )
    
    parser.add_argument(
        '-p', '--pattern',
        type=str,
        default='*.txt',
        help='文件匹配模式（默认: *.txt）'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='演练模式，不实际修改文件，仅显示统计信息'
    )
    
    args = parser.parse_args()
    
    try:
        deduplicator = DictionaryDeduplicator(directory=args.directory)
        deduplicator.process_directory(pattern=args.pattern, dry_run=args.dry_run)
    except (ValueError, IOError, OSError) as e:
        logger.error("执行失败: %s", e)
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

