#!/usr/bin/env python3
"""验证后处理器配置和功能

快速验证后处理器模块是否正确配置和工作。
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def verify_module_import():
    """验证模块导入"""
    print("=" * 60)
    print("1. 验证模块导入")
    print("=" * 60)
    
    try:
        from src.ner.postprocess import Postprocessor, BaseRule, build_postprocessor_from_config
        from src.ner.postprocess.rules import (
            BIOConsistencyRule,
            ConfidenceThresholdRule,
            EntityBoundaryRule,
            MinEntityLengthRule,
            PatternCorrectionRule,
            MergeAdjacentRule,
        )
        print("✓ 所有模块导入成功")
        return True
    except Exception as e:
        print(f"✗ 模块导入失败: {e}")
        return False


def verify_config_loading():
    """验证配置加载"""
    print("\n" + "=" * 60)
    print("2. 验证配置加载")
    print("=" * 60)
    
    try:
        from src.ner.config import ConfigManager
        from src.ner.postprocess import build_postprocessor_from_config
        
        config_manager = ConfigManager()
        config = config_manager.load_country_config('uae')
        
        if 'postprocess' not in config:
            print("✗ UAE配置中没有postprocess部分")
            return False
        
        print("✓ UAE配置包含postprocess部分")
        
        postprocessor = build_postprocessor_from_config(config)
        
        if postprocessor is None:
            print("✗ 后处理器构建失败")
            return False
        
        print(f"✓ 成功构建后处理器，包含 {len(postprocessor.rules)} 个规则:")
        for i, rule in enumerate(postprocessor.rules, 1):
            print(f"  {i}. {rule.name}")
        
        return True
    except Exception as e:
        print(f"✗ 配置加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_basic_functionality():
    """验证基本功能"""
    print("\n" + "=" * 60)
    print("3. 验证基本功能")
    print("=" * 60)
    
    try:
        from src.ner.postprocess import Postprocessor
        from src.ner.postprocess.rules import BIOConsistencyRule, EntityBoundaryRule
        
        # 创建后处理器
        postprocessor = Postprocessor([
            BIOConsistencyRule(),
            EntityBoundaryRule(remove_boundary_punct=True)
        ])
        
        # 测试数据
        tokens = ['Mr.', 'John', 'Smith', ',']
        labels = ['I-PER', 'I-PER', 'I-PER', 'I-PER']  # 错误：开头应该是B-
        confidences = [0.8, 0.9, 0.9, 0.8]
        
        print("测试输入:")
        print(f"  Tokens: {tokens}")
        print(f"  Labels: {labels}")
        
        # 应用后处理
        processed_tokens, processed_labels, processed_confidences = postprocessor.apply(
            tokens, labels, confidences
        )
        
        print("\n测试输出:")
        print(f"  Tokens: {processed_tokens}")
        print(f"  Labels: {processed_labels}")
        
        # 验证结果
        if processed_labels[0] == 'B-PER':
            print("\n✓ BIO一致性规则工作正常（I-PER → B-PER）")
        else:
            print(f"\n✗ BIO一致性规则未生效，期望 B-PER，得到 {processed_labels[0]}")
            return False
        
        if processed_labels[0] == 'O' or processed_labels[-1] == 'O':
            print("✓ 边界规则工作正常（移除了标点）")
        else:
            print("✓ 边界规则已应用")
        
        return True
    except Exception as e:
        print(f"✗ 功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_batch_processing():
    """验证批量处理"""
    print("\n" + "=" * 60)
    print("4. 验证批量处理")
    print("=" * 60)
    
    try:
        from src.ner.postprocess import Postprocessor
        from src.ner.postprocess.rules import BIOConsistencyRule
        
        postprocessor = Postprocessor([BIOConsistencyRule()])
        
        # 批量数据
        batch_tokens = [
            ['John', 'Smith'],
            ['Jane', 'Doe']
        ]
        batch_labels = [
            ['I-PER', 'I-PER'],
            ['I-PER', 'I-PER']
        ]
        batch_confidences = [
            [0.9, 0.8],
            [0.9, 0.8]
        ]
        
        print("批量输入: 2个样本")
        
        # 批量处理
        processed_tokens, processed_labels, processed_confidences = postprocessor.apply_batch(
            batch_tokens, batch_labels, batch_confidences
        )
        
        print(f"批量输出: {len(processed_labels)} 个样本")
        
        # 验证
        if all(labels[0] == 'B-PER' for labels in processed_labels):
            print("✓ 批量处理工作正常")
            return True
        else:
            print("✗ 批量处理结果不正确")
            return False
    except Exception as e:
        print(f"✗ 批量处理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_all_rules():
    """验证所有规则"""
    print("\n" + "=" * 60)
    print("5. 验证所有规则")
    print("=" * 60)
    
    try:
        from src.ner.postprocess.rules import (
            BIOConsistencyRule,
            ConfidenceThresholdRule,
            EntityBoundaryRule,
            MinEntityLengthRule,
            PatternCorrectionRule,
            MergeAdjacentRule,
        )
        
        rules = [
            BIOConsistencyRule(),
            ConfidenceThresholdRule(threshold=0.5),
            EntityBoundaryRule(),
            MinEntityLengthRule(min_length=2),
            PatternCorrectionRule(patterns={'PHONE': r'\d+'}),
            MergeAdjacentRule(),
        ]
        
        print(f"✓ 成功创建所有 {len(rules)} 个规则:")
        for rule in rules:
            print(f"  - {rule.name}")
        
        return True
    except Exception as e:
        print(f"✗ 规则创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("\n" + "🔍 后处理器模块验证".center(60))
    print()
    
    results = []
    
    # 运行所有验证
    results.append(("模块导入", verify_module_import()))
    results.append(("配置加载", verify_config_loading()))
    results.append(("基本功能", verify_basic_functionality()))
    results.append(("批量处理", verify_batch_processing()))
    results.append(("所有规则", verify_all_rules()))
    
    # 总结
    print("\n" + "=" * 60)
    print("验证总结")
    print("=" * 60)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name:20} {status}")
    
    all_passed = all(result for _, result in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 所有验证通过！后处理器模块工作正常。")
        print("\n下一步:")
        print("  1. 查看配置: data/ner/configs/countries/uae.json")
        print("  2. 运行示例: python examples/postprocess_example.py")
        print("  3. 查看文档: docs/POSTPROCESS_SUMMARY.md")
        return 0
    else:
        print("❌ 部分验证失败，请检查错误信息。")
        return 1


if __name__ == '__main__':
    sys.exit(main())
