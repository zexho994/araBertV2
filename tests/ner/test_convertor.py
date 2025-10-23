"""
测试 CSVAnnotationConvert 的优先级和多值实体处理
"""
import json
from pathlib import Path

import pandas as pd
import pytest

from src.ner.data.convertor import CSVAnnotationConvert, CSVConvertConfig


class TestCSVAnnotationConvertPriority:
    """测试实体优先级和多值处理"""

    @pytest.fixture
    def converter(self, tmp_path):
        """创建转换器实例和临时配置目录"""
        # 创建临时配置目录
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        countries_dir = config_dir / "countries"
        countries_dir.mkdir()
        
        # 创建 UAE 配置文件
        uae_config = {
            "model": {
                "base_model": "aubmindlab/bert-base-arabertv2"
            },
            "labels": {
                "entities": ["BUILDING", "STREET", "CITY", "EMIRATE", "COUNTRY"]
            },
            "data": {
                "max_length": 128
            }
        }
        
        with open(countries_dir / "uae.json", "w", encoding="utf-8") as f:
            json.dump(uae_config, f, ensure_ascii=False, indent=2)
        
        return CSVAnnotationConvert(config_dir=str(config_dir))

    def test_priority_higher_entity_wins(self, converter, tmp_path):
        """
        测试场景1：不同实体值有交集时，优先级高的实体优先标注
        
        示例：
        - formatted_address: "Dubai Tower, Dubai"
        - building: "Dubai Tower" (优先级100)
        - city: "Dubai" (优先级50)
        
        预期：building 的 "Dubai Tower" 优先被标注，
              city 的 "Dubai" 只标注第二个独立的 "Dubai"
        """
        # 创建测试 CSV
        csv_data = pd.DataFrame([
            {
                "formatted_address": "Dubai Tower, Dubai",
                "building": "Dubai Tower",
                "city": "Dubai"
            }
        ])
        csv_path = tmp_path / "test_priority.csv"
        csv_data.to_csv(csv_path, index=False)
        
        # 执行转换
        output_path = tmp_path / "output.jsonl"
        cfg = CSVConvertConfig(
            csv_path=str(csv_path),
            country_code="uae",
            output_file=str(output_path),
            validate_text_contains_entities=False
        )
        converter.generate(cfg)
        
        # 读取结果
        with open(output_path, "r", encoding="utf-8") as f:
            result = json.loads(f.readline())
        
        tokens = result["tokens"]
        labels = result["labels"]
        
        # 验证：第一个 "Dubai" 应该是 BUILDING 的一部分
        dubai_indices = [i for i, t in enumerate(tokens) if "Dubai" in t]
        assert len(dubai_indices) >= 1
        
        # 第一个 "Dubai" 应该是 B-BUILDING
        first_dubai_idx = dubai_indices[0]
        assert labels[first_dubai_idx] == "B-BUILDING", \
            f"第一个 'Dubai' 应该被标注为 B-BUILDING，实际为 {labels[first_dubai_idx]}"
        
        # "Tower" 应该是 I-BUILDING
        tower_idx = next((i for i, t in enumerate(tokens) if "Tower" in t), None)
        if tower_idx is not None:
            assert labels[tower_idx] == "I-BUILDING", \
                f"'Tower' 应该被标注为 I-BUILDING，实际为 {labels[tower_idx]}"
        
        print(f"测试通过：tokens={tokens}, labels={labels}")

    def test_longer_value_priority_in_same_entity(self, converter, tmp_path):
        """
        测试场景2：同一实体有多个值，长值优先标注
        
        示例：
        - formatted_address: "khan building and khan mall"
        - building: "khan | khan building"
        
        预期："khan building" 先被标注，剩余的 "khan" 再标注
        """
        # 创建测试 CSV
        csv_data = pd.DataFrame([
            {
                "formatted_address": "khan building and khan mall",
                "building": "khan | khan building"
            }
        ])
        csv_path = tmp_path / "test_length.csv"
        csv_data.to_csv(csv_path, index=False)
        
        # 执行转换
        output_path = tmp_path / "output.jsonl"
        cfg = CSVConvertConfig(
            csv_path=str(csv_path),
            country_code="uae",
            output_file=str(output_path),
            validate_text_contains_entities=False
        )
        converter.generate(cfg)
        
        # 读取结果
        with open(output_path, "r", encoding="utf-8") as f:
            result = json.loads(f.readline())
        
        tokens = result["tokens"]
        labels = result["labels"]
        
        # 找到 "khan" 和 "building" 的位置
        khan_indices = [i for i, t in enumerate(tokens) if "khan" in t.lower()]
        building_indices = [i for i, t in enumerate(tokens) if "building" in t.lower()]
        
        assert len(khan_indices) >= 2, "应该找到至少2个 'khan'"
        assert len(building_indices) >= 1, "应该找到至少1个 'building'"
        
        # 第一个 "khan" 应该是 B-BUILDING（因为 "khan building" 被优先标注）
        first_khan_idx = khan_indices[0]
        assert labels[first_khan_idx] == "B-BUILDING", \
            f"第一个 'khan' 应该是 B-BUILDING，实际为 {labels[first_khan_idx]}"
        
        # 紧跟的 "building" 应该是 I-BUILDING
        if building_indices:
            building_idx = building_indices[0]
            assert labels[building_idx] == "I-BUILDING", \
                f"'building' 应该是 I-BUILDING，实际为 {labels[building_idx]}"
        
        # 第二个 "khan" 也应该是 B-BUILDING
        if len(khan_indices) >= 2:
            second_khan_idx = khan_indices[1]
            assert labels[second_khan_idx] == "B-BUILDING", \
                f"第二个 'khan' 应该是 B-BUILDING，实际为 {labels[second_khan_idx]}"
        
        print(f"测试通过：tokens={tokens}, labels={labels}")

    def test_complex_priority_scenario(self, converter, tmp_path):
        """
        测试场景3：复杂场景 - 多实体交叉 + 多值
        
        示例：
        - formatted_address: "Dubai Mall, Sheikh Zayed Road, Dubai"
        - building: "Dubai Mall"
        - street: "Sheikh Zayed Road | Zayed Road"
        - city: "Dubai"
        
        预期：
        1. BUILDING 优先级最高，"Dubai Mall" 被标注为 BUILDING
        2. STREET 次之，"Sheikh Zayed Road" 被标注（长值优先）
        3. CITY 最低，最后的 "Dubai" 被标注
        """
        csv_data = pd.DataFrame([
            {
                "formatted_address": "Dubai Mall, Sheikh Zayed Road, Dubai",
                "building": "Dubai Mall",
                "street": "Sheikh Zayed Road | Zayed Road",
                "city": "Dubai"
            }
        ])
        csv_path = tmp_path / "test_complex.csv"
        csv_data.to_csv(csv_path, index=False)
        
        output_path = tmp_path / "output.jsonl"
        cfg = CSVConvertConfig(
            csv_path=str(csv_path),
            country_code="uae",
            output_file=str(output_path),
            validate_text_contains_entities=False
        )
        converter.generate(cfg)
        
        with open(output_path, "r", encoding="utf-8") as f:
            result = json.loads(f.readline())
        
        tokens = result["tokens"]
        labels = result["labels"]
        
        # 构建 token -> label 的映射（用于调试）
        token_label_pairs = list(zip(tokens, labels))
        print(f"Token-Label pairs: {token_label_pairs}")
        
        # 验证 "Dubai Mall" 被标注为 BUILDING
        dubai_mall_found = False
        for i, token in enumerate(tokens):
            if "Dubai" in token and i + 1 < len(tokens) and "Mall" in tokens[i + 1]:
                assert labels[i].startswith("B-BUILDING"), \
                    f"'Dubai' (在 Dubai Mall 中) 应该是 B-BUILDING，实际为 {labels[i]}"
                assert labels[i + 1].startswith("I-BUILDING"), \
                    f"'Mall' 应该是 I-BUILDING，实际为 {labels[i + 1]}"
                dubai_mall_found = True
                break
        
        assert dubai_mall_found, "应该找到并正确标注 'Dubai Mall'"
        
        # 验证 "Sheikh Zayed Road" 被标注为 STREET
        sheikh_found = any("Sheikh" in token or "sheikh" in token for token in tokens)
        if sheikh_found:
            sheikh_idx = next(i for i, t in enumerate(tokens) if "Sheikh" in t or "sheikh" in t)
            assert labels[sheikh_idx] == "B-STREET", \
                f"'Sheikh' 应该是 B-STREET，实际为 {labels[sheikh_idx]}"
        
        print("测试通过：复杂场景验证成功")

    def test_priority_configuration_respected(self, converter):
        """
        测试场景4：验证 _entity_priority 配置被正确使用
        
        验证优先级顺序：
        BUILDING(100) > STREET(90) > CITY(50) > COUNTRY(40)
        """
        # 验证优先级配置
        priority_map = converter._entity_priority
        
        assert priority_map.get("BUILDING", 0) > priority_map.get("STREET", 0), \
            "BUILDING 优先级应该高于 STREET"
        
        assert priority_map.get("STREET", 0) > priority_map.get("CITY", 0), \
            "STREET 优先级应该高于 CITY"
        
        assert priority_map.get("CITY", 0) > priority_map.get("COUNTRY", 0), \
            "CITY 优先级应该高于 COUNTRY"
        
        print(f"优先级配置正确：{priority_map}")

    def test_multiple_values_sorted_by_length(self, converter, tmp_path):
        """
        测试场景5：验证多值按长度降序排序
        
        示例：
        - formatted_address: "a b c d"
        - building: "a | abc | ab"
        
        预期顺序：abc (3) -> ab (2) -> a (1)
        """
        csv_data = pd.DataFrame([
            {
                "formatted_address": "abc street and ab road and a building",
                "building": "a | abc | ab"
            }
        ])
        csv_path = tmp_path / "test_sort.csv"
        csv_data.to_csv(csv_path, index=False)
        
        output_path = tmp_path / "output.jsonl"
        cfg = CSVConvertConfig(
            csv_path=str(csv_path),
            country_code="uae",
            output_file=str(output_path),
            validate_text_contains_entities=False
        )
        converter.generate(cfg)
        
        with open(output_path, "r", encoding="utf-8") as f:
            result = json.loads(f.readline())
        
        tokens = result["tokens"]
        labels = result["labels"]
        
        # 验证 "abc" 被标注（最长的）
        abc_idx = next((i for i, t in enumerate(tokens) if t == "abc"), None)
        if abc_idx is not None:
            assert labels[abc_idx] == "B-BUILDING", \
                f"'abc' 应该被标注为 B-BUILDING，实际为 {labels[abc_idx]}"
        
        # 验证 "ab" 被标注（第二长的）
        ab_idx = next((i for i, t in enumerate(tokens) if t == "ab"), None)
        if ab_idx is not None:
            assert labels[ab_idx] == "B-BUILDING", \
                f"'ab' 应该被标注为 B-BUILDING，实际为 {labels[ab_idx]}"
        
        # 验证 "a" 被标注（最短的）
        a_idx = next((i for i, t in enumerate(tokens) if t == "a"), None)
        if a_idx is not None:
            assert labels[a_idx] == "B-BUILDING", \
                f"'a' 应该被标注为 B-BUILDING，实际为 {labels[a_idx]}"
        
        print("测试通过：多值按长度排序正确")

    def test_no_overlap_when_higher_priority_tagged_first(self, converter, tmp_path):
        """
        测试场景6：验证高优先级实体先标注后，低优先级实体不会覆盖
        
        示例：
        - formatted_address: "Dubai Tower in Dubai"
        - building: "Dubai Tower" (优先级100)
        - emirate: "Dubai" (优先级60)
        
        预期：第一个 "Dubai" 属于 BUILDING，不应该被 EMIRATE 覆盖
        """
        csv_data = pd.DataFrame([
            {
                "formatted_address": "Dubai Tower in Dubai city",
                "building": "Dubai Tower",
                "emirate": "Dubai"
            }
        ])
        csv_path = tmp_path / "test_no_overlap.csv"
        csv_data.to_csv(csv_path, index=False)
        
        output_path = tmp_path / "output.jsonl"
        cfg = CSVConvertConfig(
            csv_path=str(csv_path),
            country_code="uae",
            output_file=str(output_path),
            validate_text_contains_entities=False
        )
        converter.generate(cfg)
        
        with open(output_path, "r", encoding="utf-8") as f:
            result = json.loads(f.readline())
        
        tokens = result["tokens"]
        labels = result["labels"]
        
        # 找到所有 "Dubai" 的位置
        dubai_indices = [i for i, t in enumerate(tokens) if "Dubai" in t]
        assert len(dubai_indices) >= 2, "应该找到至少2个 'Dubai'"
        
        # 第一个 "Dubai" 应该是 BUILDING，不是 EMIRATE
        first_dubai_idx = dubai_indices[0]
        assert labels[first_dubai_idx] == "B-BUILDING", \
            f"第一个 'Dubai' 应该保持为 B-BUILDING，不应被 EMIRATE 覆盖，实际为 {labels[first_dubai_idx]}"
        
        # 第二个 "Dubai" 应该是 EMIRATE
        second_dubai_idx = dubai_indices[1]
        assert labels[second_dubai_idx] == "B-EMIRATE", \
            f"第二个 'Dubai' 应该是 B-EMIRATE，实际为 {labels[second_dubai_idx]}"
        
        print("测试通过：无重叠覆盖")


def test_entity_priority_values():
    """独立测试：验证实体优先级数值配置"""
    converter = CSVAnnotationConvert()
    
    priority = converter._entity_priority
    
    # 验证关键实体的优先级顺序
    assert priority["BUILDING"] == 100, "BUILDING 优先级应该是 100"
    assert priority["STREET"] == 90, "STREET 优先级应该是 90"
    assert priority["COMPOUND"] == 80, "COMPOUND 优先级应该是 80"
    assert priority["SUB_AREA"] == 70, "SUB_AREA 优先级应该是 70"
    assert priority["EMIRATE"] == 60, "EMIRATE 优先级应该是 60"
    assert priority["CITY"] == 50, "CITY 优先级应该是 50"
    assert priority["COUNTRY"] == 40, "COUNTRY 优先级应该是 40"
    
    print(f"实体优先级配置正确：{priority}")


if __name__ == "__main__":
    # 本地运行测试
    import sys
    
    # 添加项目根目录到 sys.path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    # 运行独立测试
    print("=" * 60)
    print("运行独立测试：实体优先级值")
    print("=" * 60)
    test_entity_priority_values()
    
    print("\n" + "=" * 60)
    print("运行类测试需要使用 pytest:")
    print("pytest tests/ner/test_convertor.py -v")
    print("=" * 60)

