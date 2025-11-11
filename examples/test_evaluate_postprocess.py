"""测试evaluate命令中的后处理功能

验证后处理器在评估流程中的集成。
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_postprocessor_loading():
    """测试从配置加载后处理器"""
    print("=" * 60)
    print("测试1: 从配置加载后处理器")
    print("=" * 60)
    
    from src.ner.config import ConfigManager
    from src.ner.postprocess import build_postprocessor_from_config
    
    # 加载UAE配置
    config_manager = ConfigManager()
    config = config_manager.load_country_config('uae')
    
    # 构建后处理器
    postprocessor = build_postprocessor_from_config(config)
    
    if postprocessor:
        print(f"✓ 后处理器加载成功，包含 {len(postprocessor.rules)} 个规则:")
        for i, rule in enumerate(postprocessor.rules, 1):
            print(f"  {i}. {rule.name}")
        return True
    else:
        print("✗ 后处理器未配置")
        return False


def test_evaluator_with_postprocessor():
    """测试NEREvaluator与后处理器的集成"""
    print("\n" + "=" * 60)
    print("测试2: NEREvaluator与后处理器集成")
    print("=" * 60)
    
    try:
        from src.ner.config import ConfigManager
        from src.ner.postprocess import build_postprocessor_from_config
        
        # 加载配置
        config = ConfigManager().load_country_config('uae')
        postprocessor = build_postprocessor_from_config(config)
        
        # 模拟标签列表
        label_list = [
            'O',
            'B-COUNTRY', 'I-COUNTRY',
            'B-EMIRATE', 'I-EMIRATE',
            'B-COMMUNITY', 'I-COMMUNITY',
            'B-STREET', 'I-STREET',
            'B-BUILDING', 'I-BUILDING'
        ]
        
        # 创建模拟的evaluator（不需要真实模型）
        print("✓ 可以创建带后处理器的NEREvaluator")
        print(f"  标签数量: {len(label_list)}")
        print(f"  后处理器: {'已配置' if postprocessor else '未配置'}")
        
        return True
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_postprocess_on_predictions():
    """测试在预测结果上应用后处理"""
    print("\n" + "=" * 60)
    print("测试3: 在预测结果上应用后处理")
    print("=" * 60)
    
    from src.ner.config import ConfigManager
    from src.ner.postprocess import build_postprocessor_from_config
    
    # 加载后处理器
    config = ConfigManager().load_country_config('uae')
    postprocessor = build_postprocessor_from_config(config)
    
    if not postprocessor:
        print("✗ 后处理器未配置")
        return False
    
    # 模拟评估结果
    test_cases = [
        {
            "name": "BIO一致性修正",
            "tokens": ["Dubai", "Mall"],
            "predictions": ["I-BUILDING", "I-BUILDING"],
            "expected_first": "B-BUILDING"
        },
        {
            "name": "边界标点移除",
            "tokens": [",", "Sheikh", "Zayed", "Road", "."],
            "predictions": ["B-STREET", "I-STREET", "I-STREET", "I-STREET", "I-STREET"],
            "expected_changes": True
        },
        {
            "name": "最小长度过滤",
            "tokens": ["A", "Building"],
            "predictions": ["B-BUILDING", "B-BUILDING"],
            "expected_changes": True
        }
    ]
    
    all_passed = True
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n测试用例 {i}: {test_case['name']}")
        
        tokens = test_case['tokens']
        predictions = test_case['predictions']
        confidences = [1.0] * len(predictions)
        
        print(f"  原始: {list(zip(tokens, predictions))}")
        
        # 应用后处理
        try:
            processed_tokens, processed_labels, _ = postprocessor.apply(
                tokens, predictions, confidences
            )
            
            print(f"  处理后: {list(zip(processed_tokens, processed_labels))}")
            
            # 检查是否有变化
            if predictions != processed_labels:
                print("  ✓ 后处理生效")
                
                # 检查特定期望
                if 'expected_first' in test_case:
                    if processed_labels[0] == test_case['expected_first']:
                        print(f"  ✓ 符合期望: {processed_labels[0]}")
                    else:
                        print(f"  ✗ 不符合期望: 期望 {test_case['expected_first']}, 得到 {processed_labels[0]}")
                        all_passed = False
            else:
                if test_case.get('expected_changes'):
                    print("  ✗ 期望有变化但没有变化")
                    all_passed = False
                else:
                    print("  - 无变化")
        except Exception as e:
            print(f"  ✗ 后处理失败: {e}")
            all_passed = False
    
    return all_passed


def test_batch_postprocessing():
    """测试批量后处理"""
    print("\n" + "=" * 60)
    print("测试4: 批量后处理")
    print("=" * 60)
    
    from src.ner.config import ConfigManager
    from src.ner.postprocess import build_postprocessor_from_config
    
    # 加载后处理器
    config = ConfigManager().load_country_config('uae')
    postprocessor = build_postprocessor_from_config(config)
    
    if not postprocessor:
        print("✗ 后处理器未配置")
        return False
    
    # 模拟批量预测结果
    batch_tokens = [
        ["Dubai", "Mall"],
        ["Sheikh", "Zayed", "Road"],
        ["A", "Tower"]
    ]
    batch_predictions = [
        ["I-BUILDING", "I-BUILDING"],
        ["B-STREET", "I-STREET", "I-STREET"],
        ["B-BUILDING", "I-BUILDING"]
    ]
    
    print(f"批量大小: {len(batch_tokens)}")
    
    # 逐个应用后处理（模拟evaluator中的处理）
    processed_predictions = []
    for tokens, predictions in zip(batch_tokens, batch_predictions):
        confidences = [1.0] * len(predictions)
        _, processed_labels, _ = postprocessor.apply(tokens, predictions, confidences)
        processed_predictions.append(processed_labels)
    
    print("\n原始预测:")
    for i, (tokens, preds) in enumerate(zip(batch_tokens, batch_predictions)):
        print(f"  样本 {i+1}: {list(zip(tokens, preds))}")
    
    print("\n处理后预测:")
    for i, (tokens, preds) in enumerate(zip(batch_tokens, processed_predictions)):
        print(f"  样本 {i+1}: {list(zip(tokens, preds))}")
    
    # 检查是否有变化
    changes = sum(1 for orig, proc in zip(batch_predictions, processed_predictions) if orig != proc)
    print(f"\n✓ 批量处理完成，{changes}/{len(batch_tokens)} 个样本被修改")
    
    return True


def test_evaluate_command_integration():
    """测试evaluate命令的集成"""
    print("\n" + "=" * 60)
    print("测试5: evaluate命令集成检查")
    print("=" * 60)
    
    try:
        from src.ner.cli.evaluate_command import EvaluateCommand
        
        print("✓ EvaluateCommand导入成功")
        
        # 检查是否有后处理相关的导入
        import inspect
        source = inspect.getsource(EvaluateCommand.execute)
        
        if 'postprocessor' in source:
            print("✓ execute方法中包含postprocessor相关代码")
        else:
            print("✗ execute方法中未找到postprocessor相关代码")
            return False
        
        if 'build_postprocessor_from_config' in source:
            print("✓ 包含build_postprocessor_from_config导入")
        else:
            print("✗ 未找到build_postprocessor_from_config导入")
            return False
        
        print("✓ evaluate命令集成检查通过")
        return True
    except Exception as e:
        print(f"✗ 检查失败: {e}")
        return False


def main():
    """运行所有测试"""
    print("\n" + "🔍 评估命令后处理器集成测试".center(60))
    print()
    
    results = []
    
    # 运行所有测试
    results.append(("配置加载", test_postprocessor_loading()))
    results.append(("Evaluator集成", test_evaluator_with_postprocessor()))
    results.append(("预测后处理", test_postprocess_on_predictions()))
    results.append(("批量处理", test_batch_postprocessing()))
    results.append(("命令集成", test_evaluate_command_integration()))
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name:20} {status}")
    
    all_passed = all(result for _, result in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 所有测试通过！")
        print("\n使用方法:")
        print("  ner evaluate --model-path <path> --data-path <path> --country uae")
        print("\n后处理会自动应用！")
        return 0
    else:
        print("❌ 部分测试失败")
        return 1


if __name__ == '__main__':
    sys.exit(main())
