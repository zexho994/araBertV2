#!/usr/bin/env python3
"""
简单的调试脚本 - 用于在 IDE 中调试 NER CLI 的训练与评估命令

使用方法：
1. 在 IDE 中打开此文件
2. 在需要调试的地方设置断点
3. 运行此脚本进行调试

示例命令：
- 训练：python .\ner_cli.py train --country uae_xml_roberta_base
- 评估：python .\ner_cli.py evaluate --country uae_xml_roberta_base --model-path <path> --data-path <path>

# TODO: 从 `@configs` 自动解析默认数据/模型路径，减少硬编码依赖。
"""

import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def exec_train_command(country : str):
    """调试训练命令

    说明：
    - 通过直接设置 `sys.argv` 模拟命令行参数，便于在 IDE 中断点调试。
    - 会切换到项目根目录，确保相对路径解析与实际运行一致。

    # TODO: 支持从环境变量或配置文件中读取待调试的参数，避免改动源码。
    """
    
    # 模拟命令行参数
    # 原命令: python .\ner_cli.py train --country uae_xml_roberta_base
    sys.argv = [
        'ner_cli.py',
        '--verbose',
        'train', 
        '--country', 
        country,
    ]
    
    print(f"模拟命令行参数: {' '.join(sys.argv)}")
    print(f"当前工作目录: {os.getcwd()}")
    print(f"项目根目录: {project_root}")
    
    # 切换到项目根目录，确保相对路径与 CLI 一致
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

def exec_evaluate_command(country : str = None):
    """调试评估命令

    说明：
    - 同样通过设置 `sys.argv` 模拟评估子命令。
    - 默认提供示例模型与数据路径，实际使用请按需修改。

    # TODO: 验证 `--data-path` 的默认行为是否与 CLI 说明一致，并在缺省时从配置中解析。
    """
    sys.argv = [
        'ner_cli.py',
        '--verbose',
        'evaluate',
        '--country',
        country,
        '--model-path',
        f'data/ner/models/{country}/best_model',
        '--data-path',
        f'data/ner/data/{country}/val.jsonl',
    ]
    print(f"模拟命令行参数: {' '.join(sys.argv)}")
    print(f"当前工作目录: {os.getcwd()}")
    print(f"项目根目录: {project_root}")
    
    os.chdir(project_root)

    try:
        import ner_cli
        print("开始执行评估命令...")
        print("=" * 50)

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

    country = 'uae_xml_roberta_base'
    
    # 默认调试评估命令，若需训练请改为调用 `debug_train_command()`
    # TODO: 通过命令行/环境变量选择调试目标（train/evaluate）。
    exec_train_command(country = country)
    exec_evaluate_command(country = country)
    
    print(f"\n脚本执行完成")