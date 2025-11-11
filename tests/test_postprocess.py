"""后处理模块测试

验证后处理规则的正确性。
"""

import pytest
from src.ner.postprocess import Postprocessor, BaseRule
from src.ner.postprocess.rules import (
    BIOConsistencyRule,
    ConfidenceThresholdRule,
    EntityBoundaryRule,
    MinEntityLengthRule,
    PatternCorrectionRule
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


class TestEntityBoundaryRule:
    """测试实体边界规则"""
    
    def test_remove_boundary_punct(self):
        """测试移除边界标点"""
        rule = EntityBoundaryRule(remove_boundary_punct=True)
        
        tokens = ['Mr.', 'John', 'Smith', ',']
        labels = ['B-PER', 'I-PER', 'I-PER', 'I-PER']
        confidences = [0.9, 0.9, 0.9, 0.9]
        
        from src.ner.postprocess.pipeline import PredictionResult
        result = PredictionResult(tokens, labels, confidences)
        processed = rule.apply(result)
        
        assert processed.labels[0] == 'O'  # Mr. 被移除
        assert processed.labels[1] == 'B-PER'
        assert processed.labels[2] == 'I-PER'
        assert processed.labels[3] == 'O'  # , 被移除


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
        
        processed_tokens, processed_labels, processed_confidences = postprocessor.apply(
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
        
        processed_tokens, processed_labels, processed_confidences = postprocessor.apply_batch(
            batch_tokens, batch_labels, batch_confidences
        )
        
        assert processed_labels[0][0] == 'B-PER'
        assert processed_labels[1][0] == 'B-PER'


class TestBuilder:
    """测试配置构建器"""
    
    def test_build_from_config(self):
        """测试从配置构建"""
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
                    }
                ]
            }
        }
        
        postprocessor = build_postprocessor_from_config(config)
        
        assert len(postprocessor.rules) == 2
        assert isinstance(postprocessor.rules[0], BIOConsistencyRule)
        assert isinstance(postprocessor.rules[1], ConfidenceThresholdRule)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
