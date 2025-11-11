"""测试REPL中的后处理功能

模拟REPL中的后处理流程。
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_postprocessor_in_repl():
    """测试在REPL中使用后处理器"""
    print("=" * 60)
    print("测试REPL中的后处理功能")
    print("=" * 60)
    
    from src.ner.config import ConfigManager
    from src.ner.postprocess import build_postprocessor_from_config
    
    # 模拟REPL加载国家配置
    print("\n1. 加载国家配置...")
    config_manager = ConfigManager()
    config = config_manager.load_country_config('uae')
    print("✓ 配置加载成功")
    
    # 构建后处理器
    print("\n2. 构建后处理器...")
    postprocessor = build_postprocessor_from_config(config)
    
    if postprocessor:
        print(f"✓ 后处理器构建成功，包含 {len(postprocessor.rules)} 个规则:")
        for i, rule in enumerate(postprocessor.rules, 1):
            print(f"   {i}. {rule.name}")
    else:
        print("✗ 未配置后处理器")
        return
    
    # 模拟预测结果
    print("\n3. 模拟预测结果...")
    prediction = {
        "text": "Mr. Ahmed lives at A Tower , Dubai",
        "tokens": ["Mr.", "Ahmed", "lives", "at", "A", "Tower", ",", "Dubai"],
        "labels": ["I-PER", "I-PER", "O", "O", "B-BUILDING", "I-BUILDING", "I-BUILDING", "B-EMIRATE"],
        "confidences": [0.7, 0.9, 0.95, 0.95, 0.6, 0.9, 0.8, 0.9],
        "entities": []
    }
    
    print("原始预测:")
    for t, l, c in zip(prediction["tokens"], prediction["labels"], prediction["confidences"]):
        print(f"  {t:15} {l:15} {c:.2f}")
    
    # 应用后处理
    print("\n4. 应用后处理...")
    tokens = prediction["tokens"]
    labels = prediction["labels"]
    confidences = prediction["confidences"]
    
    processed_tokens, processed_labels, processed_confidences = postprocessor.apply(
        tokens, labels, confidences
    )
    
    print("后处理结果:")
    for t, l, c in zip(processed_tokens, processed_labels, processed_confidences):
        print(f"  {t:15} {l:15} {c:.2f}")
    
    # 显示变化
    print("\n5. 变化总结:")
    changes = []
    for i, (orig, new) in enumerate(zip(labels, processed_labels)):
        if orig != new:
            changes.append(f"  - {tokens[i]}: {orig} → {new}")
    
    if changes:
        print("标签变化:")
        for change in changes:
            print(change)
    else:
        print("  无变化")
    
    print("\n✓ 测试完成")


def test_repl_workflow():
    """测试完整的REPL工作流程"""
    print("\n" + "=" * 60)
    print("测试完整REPL工作流程")
    print("=" * 60)
    
    from src.ner.cli.repl_predict import PredictREPL
    
    # 创建REPL实例
    print("\n1. 创建REPL实例...")
    repl = PredictREPL()
    print("✓ REPL实例创建成功")
    
    # 模拟设置国家
    print("\n2. 设置国家配置...")
    repl._handle_country('uae')
    
    if repl.postprocessor:
        print(f"✓ 后处理器已加载，包含 {len(repl.postprocessor.rules)} 个规则")
    else:
        print("✗ 后处理器未加载")
        return
    
    # 显示信息
    print("\n3. 显示REPL状态:")
    repl._print_info()
    
    print("\n✓ 工作流程测试完成")


def test_postprocess_rules():
    """测试各个后处理规则"""
    print("\n" + "=" * 60)
    print("测试后处理规则效果")
    print("=" * 60)
    
    from src.ner.postprocess import Postprocessor
    from src.ner.postprocess.rules import (
        BIOConsistencyRule,
    )
    
    # 测试用例
    test_cases = [
        {
            "name": "BIO一致性修正",
            "rule": BIOConsistencyRule(),
            "tokens": ["Dubai", "Mall"],
            "labels": ["I-BUILDING", "I-BUILDING"],
            "confidences": [0.9, 0.9],
            "expected_change": "I-BUILDING → B-BUILDING"
        },
        {
            "name": "边界标点移除",
            "rule": EntityBoundaryRule(remove_boundary_punct=True),
            "tokens": [",", "Sheikh", "Zayed", "Road", "."],
            "labels": ["B-STREET", "I-STREET", "I-STREET", "I-STREET", "I-STREET"],
            "confidences": [0.8, 0.9, 0.9, 0.9, 0.8],
            "expected_change": "移除边界标点"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n测试 {i}: {test_case['name']}")
        print(f"期望: {test_case['expected_change']}")
        
        postprocessor = Postprocessor([test_case['rule']])
        
        print("原始:")
        for t, l in zip(test_case['tokens'], test_case['labels']):
            print(f"  {t:15} {l}")
        
        processed_tokens, processed_labels, processed_confidences = postprocessor.apply(
            test_case['tokens'],
            test_case['labels'],
            test_case['confidences']
        )
        
        print("处理后:")
        for t, l in zip(processed_tokens, processed_labels):
            print(f"  {t:15} {l}")
        
        # 检查是否有变化
        if test_case['labels'] != processed_labels:
            print("✓ 规则生效")
        else:
            print("- 无变化")


if __name__ == '__main__':
    test_postprocessor_in_repl()
    test_repl_workflow()
    test_postprocess_rules()
    
    print("\n" + "=" * 60)
    print("🎉 所有测试完成！")
    print("\n使用方法:")
    print("  1. 启动REPL: ner predict")
    print("  2. 设置国家: country uae")
    print("  3. 加载模型: load <model_path>")
    print("  4. 预测文本: predict <text>")
    print("\n后处理会自动应用！")
    print("=" * 60)
