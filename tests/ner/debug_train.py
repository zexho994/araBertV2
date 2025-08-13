#!/usr/bin/env python3
"""
简单的调试脚本 - 用于在IDE中调试 NER CLI 训练命令

使用方法：
1. 在IDE中打开此文件
2. 在需要调试的地方设置断点
3. 运行此脚本进行调试

模拟命令：python .\ner_cli.py train --country uae_xml_roberta_base
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def debug_train_command():
    """调试训练命令"""
    
    # 模拟命令行参数
    # 原命令: python .\ner_cli.py train --country uae_xml_roberta_base
    sys.argv = [
        'ner_cli.py',
        'train', 
        '--country', 
        'uae_xml_roberta_base'
    ]
    
    print(f"模拟命令行参数: {' '.join(sys.argv)}")
    print(f"当前工作目录: {os.getcwd()}")
    print(f"项目根目录: {project_root}")
    
    # 切换到项目根目录
    os.chdir(project_root)
    
    try:
        # 导入并执行 ner_cli 的主函数
        import ner_cli
        
        print("开始执行训练命令...")
        print("=" * 50)
        
        # 在这里可以设置断点进行调试
        result = ner_cli.main()
        
        print("=" * 50)
        print(f"命令执行完成，返回码: {result}")
        
        return result
        
    except Exception as e:
        print(f"执行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    print("NER CLI 训练命令调试脚本")
    print("=" * 30)
    
    # 执行调试
    exit_code = debug_train_command()
    
    print(f"\n脚本执行完成，退出码: {exit_code}")