"""后处理模块测试

验证后处理规则的正确性。
"""

import pytest
from src.ner.postprocess import Postprocessor
from src.ner.postprocess.rules import (
    BIOConsistencyRule,
    ConfidenceThresholdRule,
    RegexFilterRule,
    BlacklistFilterRule,
)
from src.ner.postprocess.builder import build_postprocessor_from_config


class TestBIOConsistencyRule:
    """测试BIO一致性规则"""
    
    def test_fix_orphan_i_to_b(self):
        """测试孤立I标签转换为B标签"""
        rule = BIOConsistencyRule(fix_orphan_i='to_b')
        
        tokens = ['John', 'Smith', 'works']
        labels = ['I-PER', 'I-PER', 'O']
        confidences = [0.9, 0.8, 0.95]
        
        from src.ner.postprocess.pipeline import PredictionResult
        result = PredictionResult(tokens, labels, confidences)
        processed = rule.apply(result)
        
        assert processed.labels[0] == 'B-PER'
        assert processed.labels[1] == 'I-PER'
    
    def test_fix_type_mismatch(self):
        """测试类型不匹配的修正"""
        rule = BIOConsistencyRule()
        
        tokens = ['John', 'Smith', 'Inc']
        labels = ['B-PER', 'I-PER', 'I-ORG']
        confidences = [0.9, 0.8, 0.7]
        
        from src.ner.postprocess.pipeline import PredictionResult
        result = PredictionResult(tokens, labels, confidences)
        processed = rule.apply(result)
        
        assert processed.labels[2] == 'B-ORG'


class TestConfidenceThresholdRule:
    """测试置信度阈值规则"""
    
    def test_simple_threshold(self):
        """测试简单阈值过滤"""
        rule = ConfidenceThresholdRule(threshold=0.7)
        
        tokens = ['John', 'Smith', 'works']
        labels = ['B-PER', 'I-PER', 'O']
        confidences = [0.9, 0.6, 0.95]
        
        from src.ner.postprocess.pipeline import PredictionResult
        result = PredictionResult(tokens, labels, confidences)
        processed = rule.apply(result)
        
        assert processed.labels[0] == 'B-PER'
        assert processed.labels[1] == 'O'  # 低于阈值

class TestPostprocessor:
    """测试后处理器主类"""
    
    def test_pipeline(self):
        """测试规则链"""
        postprocessor = Postprocessor([
            BIOConsistencyRule(),
            ConfidenceThresholdRule(threshold=0.7)
        ])
        
        tokens = ['I', 'am', 'John']
        labels = ['O', 'O', 'I-PER']
        confidences = [0.9, 0.9, 0.8]
        
        _, processed_labels, _ = postprocessor.apply(
            tokens, labels, confidences
        )
        
        assert processed_labels[2] == 'B-PER'  # 一致性规则修正
    
    def test_batch_processing(self):
        """测试批量处理"""
        postprocessor = Postprocessor([BIOConsistencyRule()])
        
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
        
        _, processed_labels, _ = postprocessor.apply_batch(
            batch_tokens, batch_labels, batch_confidences
        )
        
        assert processed_labels[0][0] == 'B-PER'
        assert processed_labels[1][0] == 'B-PER'


class TestBuilder:
    """测试配置构建器"""
    
    def test_build_from_config(self, tmp_path):
        """测试从配置构建"""
        building_file = tmp_path / "building.txt"
        street_file = tmp_path / "street.txt"
        building_file.write_text("Tower A\nTower B\n", encoding='utf-8')
        street_file.write_text("Main Street\n", encoding='utf-8')
        
        config = {
            "postprocess": {
                "rules": [
                    {
                        "type": "bio_consistency",
                        "params": {"fix_orphan_i": "to_b"}
                    },
                    {
                        "type": "confidence_threshold",
                        "params": {"threshold": 0.5}
                    },
                    {
                        "type": "regex_filter",
                        "params": {
                            "patterns": ["^PO \\d{5}$"],
                            "join_with": " "
                        }
                    },
                    {
                        "type": "blacklist_filter",
                        "params": {
                            "blacklist_files": {
                                "BUILDING": str(building_file),
                                "STREET": str(street_file),
                            },
                            "join_with": " ",
                            "case_insensitive": True
                        }
                    }
                ]
            }
        }
        
        postprocessor = build_postprocessor_from_config(config)
        
        assert len(postprocessor.rules) == 4
        assert isinstance(postprocessor.rules[0], BIOConsistencyRule)
        assert isinstance(postprocessor.rules[1], ConfidenceThresholdRule)
        assert isinstance(postprocessor.rules[2], RegexFilterRule)
        assert isinstance(postprocessor.rules[3], BlacklistFilterRule)


class TestBlacklistFilterRule:
    """测试黑名单过滤规则"""
    
    def test_blacklist_filter_rule(self, tmp_path):
        """实体命中黑名单应被清除"""
        building_file = tmp_path / "building.txt"
        building_file.write_text("Tower 42\nForbidden Plaza\n", encoding='utf-8')
        
        rule = BlacklistFilterRule(
            blacklist_files={"BUILDING": str(building_file)},
            join_with=" ",
            case_insensitive=True,
        )
        
        tokens = ['Forbidden', 'Plaza', 'is', 'closed']
        labels = ['B-BUILDING', 'I-BUILDING', 'O', 'O']
        confidences = [0.9, 0.85, 0.7, 0.6]
        
        from src.ner.postprocess.pipeline import PredictionResult
        result = PredictionResult(tokens, labels, confidences)
        processed = rule.apply(result)
        
        assert processed.labels[:2] == ['O', 'O']
        assert processed.labels[2:] == ['O', 'O']
    
    def test_blacklist_filter_rule_case_sensitive(self, tmp_path):
        """大小写敏感时按原样匹配"""
        street_file = tmp_path / "street.txt"
        street_file.write_text("King Road\n", encoding='utf-8')
        
        rule_sensitive = BlacklistFilterRule(
            blacklist_files={"STREET": str(street_file)},
            join_with=" ",
            case_insensitive=False,
        )
        rule_insensitive = BlacklistFilterRule(
            blacklist_files={"STREET": str(street_file)},
            join_with=" ",
            case_insensitive=True,
        )
        
        tokens = ['king', 'road']
        labels = ['B-STREET', 'I-STREET']
        confidences = [0.9, 0.9]
        
        from src.ner.postprocess.pipeline import PredictionResult
        result = PredictionResult(tokens, labels, confidences)
        
        processed_sensitive = rule_sensitive.apply(result)
        processed_insensitive = rule_insensitive.apply(result)
        
        assert processed_sensitive.labels == ['B-STREET', 'I-STREET']
        assert processed_insensitive.labels == ['O', 'O']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
