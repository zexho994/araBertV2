"""NER Metrics

Provides simple token-level precision, recall, F1 and accuracy for BIO labels.
"""

from __future__ import annotations

from typing import Dict, List, Sequence


class NERMetrics:
    def __init__(self, id2label: Dict[int, str]):
        self.id2label = id2label or {}

    def compute_metrics(self, predictions: Sequence[int], labels: Sequence[int]) -> Dict[str, float]:
        if not predictions or not labels:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}

        if len(predictions) != len(labels):
            raise ValueError("Predictions and labels must have the same length")

        total = len(labels)
        correct = 0

        true_positive = 0
        false_positive = 0
        false_negative = 0

        for pred_id, true_id in zip(predictions, labels):
            pred_label = self.id2label.get(int(pred_id), "O")
            true_label = self.id2label.get(int(true_id), "O")

            if pred_label == true_label:
                correct += 1

            pred_is_entity = pred_label != "O"
            true_is_entity = true_label != "O"

            if pred_is_entity and true_is_entity and pred_label == true_label:
                true_positive += 1
            elif pred_is_entity and pred_label != true_label:
                false_positive += 1
            elif true_is_entity and not pred_is_entity:
                false_negative += 1

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
        }


