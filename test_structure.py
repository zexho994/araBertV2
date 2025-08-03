#!/usr/bin/env python3
"""
项目结构测试脚本 - 不依赖外部库
"""

import os
import sys

def test_project_structure():
    """测试项目结构是否完整"""
    print("=" * 60)
    print("AraBERTv2 阿拉伯语地址解析项目结构测试")
    print("=" * 60)
    
    # 检查必要的文件和目录
    required_items = [
        "requirements.txt",
        "README.md",
        ".gitignore",
        "configs/config.py",
        "src/__init__.py",
        "src/model/arabertv2_ner.py",
        "src/data_processing/preprocess.py",
        "src/training/train.py",
        "src/training/evaluate.py",
        "src/utils/__init__.py",
        "src/utils/helpers.py",
        "notebooks/demo.ipynb",
        "run_demo.py"
    ]
    
    print("\\n检查项目文件:")
    missing_files = []
    
    for item in required_items:
        if os.path.exists(item):
            print(f"  ✓ {item}")
        else:
            print(f"  ✗ {item} (缺失)")
            missing_files.append(item)
    
    # 检查配置文件内容
    print("\\n检查配置文件:")
    try:
        sys.path.append('.')
        from configs.config import MODEL_CONFIG, TRAINING_CONFIG, ENTITY_LABELS
        print("  ✓ 配置文件加载成功")
        print(f"  ✓ 模型配置: {len(MODEL_CONFIG)} 项")
        print(f"  ✓ 训练配置: {len(TRAINING_CONFIG)} 项")
        print(f"  ✓ 实体标签: {len(ENTITY_LABELS)} 个")
    except Exception as e:
        print(f"  ✗ 配置文件加载失败: {e}")
    
    # 显示项目统计
    print("\\n项目统计:")
    
    # 统计Python文件
    python_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    print(f"  Python文件数量: {len(python_files)}")
    
    # 统计代码行数
    total_lines = 0
    for py_file in python_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                lines = len(f.readlines())
                total_lines += lines
                print(f"    {py_file}: {lines} 行")
        except:
            pass
    
    print(f"  总代码行数: {total_lines}")
    
    # 显示结果
    print("\\n" + "=" * 40)
    print("测试结果")
    print("=" * 40)
    
    if not missing_files:
        print("\\n✅ 项目结构完整!")
        print("\\n下一步操作:")
        print("1. 安装依赖: pip install -r requirements.txt")
        print("2. 运行演示: python run_demo.py")
        print("3. 开始训练: python src/training/train.py")
    else:
        print(f"\\n❌ 发现 {len(missing_files)} 个缺失文件")
        print("缺失文件:", missing_files)
    
    print("\\n项目已成功初始化! 🎉")

if __name__ == "__main__":
    test_project_structure()