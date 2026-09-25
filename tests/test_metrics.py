"""
Unit tests for evaluation metric module (src/evaluation/metrics.py).
"""

import sys
import os
import unittest
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.metrics import calculate_single_entity_f05, score_f05


class TestF05Metric(unittest.TestCase):
    
    def test_perfect_match(self):
        # 100% precision, 100% recall
        true_ids = {'S2-100', 'S3-200'}
        pred_ids = {'S2-100', 'S3-200'}
        score = calculate_single_entity_f05(true_ids, pred_ids)
        self.assertAlmostEqual(score, 1.0, places=6)

    def test_partial_recall(self):
        # 1 true positive out of 2 true matches, 0 false positives -> Precision = 1.0, Recall = 0.5
        # F0.5 = (1.25 * 1.0 * 0.5) / (0.25 * 1.0 + 0.5) = 0.625 / 0.75 = 5/6 = 0.833333...
        true_ids = {'S2-100', 'S3-200'}
        pred_ids = {'S2-100'}
        score = calculate_single_entity_f05(true_ids, pred_ids)
        self.assertAlmostEqual(score, 5.0 / 6.0, places=6)

    def test_over_prediction_false_positive(self):
        # 1 true positive, 1 false positive -> Precision = 0.5, Recall = 1.0
        # F0.5 = (1.25 * 0.5 * 1.0) / (0.25 * 0.5 + 1.0) = 0.625 / 1.125 = 5/9 = 0.555555...
        true_ids = {'S2-100'}
        pred_ids = {'S2-100', 'S2-999'}
        score = calculate_single_entity_f05(true_ids, pred_ids)
        self.assertAlmostEqual(score, 5.0 / 9.0, places=6)

    def test_singleton_correctly_empty(self):
        # True no-match entity predicted as no-match -> F0.5 = 1.0
        true_ids = set()
        pred_ids = set()
        score = calculate_single_entity_f05(true_ids, pred_ids)
        self.assertEqual(score, 1.0)

    def test_singleton_false_positive(self):
        # True no-match entity predicted with a match -> F0.5 = 0.0
        true_ids = set()
        pred_ids = {'S2-100'}
        score = calculate_single_entity_f05(true_ids, pred_ids)
        self.assertEqual(score, 0.0)

    def test_non_empty_predicted_empty(self):
        # True matches exist, but predicted empty -> F0.5 = 0.0
        true_ids = {'S2-100'}
        pred_ids = set()
        score = calculate_single_entity_f05(true_ids, pred_ids)
        self.assertEqual(score, 0.0)

    def test_completely_wrong_match(self):
        # True match is S2-100, predicted S2-999 -> Precision = 0, Recall = 0 -> F0.5 = 0.0
        true_ids = {'S2-100'}
        pred_ids = {'S2-999'}
        score = calculate_single_entity_f05(true_ids, pred_ids)
        self.assertEqual(score, 0.0)

    def test_macro_average_scoring_dict(self):
        y_true = {
            'S1-1': {'S2-10', 'S3-10'},  # score = 1.0 (exact match)
            'S1-2': set(),               # score = 1.0 (singleton correct)
            'S1-3': {'S2-30'},           # score = 0.0 (missed prediction)
            'S1-4': {'S2-40', 'S3-40'}   # score = 5/6 = 0.833333 (pred {'S2-40'})
        }
        y_pred = {
            'S1-1': {'S2-10', 'S3-10'},
            'S1-2': set(),
            'S1-3': set(),
            'S1-4': {'S2-40'}
        }
        
        expected_macro = (1.0 + 1.0 + 0.0 + (5.0 / 6.0)) / 4.0  # = 2.833333 / 4 = 0.7083333...
        macro_score = score_f05(y_true, y_pred)
        self.assertAlmostEqual(macro_score, expected_macro, places=6)

    def test_macro_average_scoring_dataframe(self):
        y_true_df = pd.DataFrame([
            {'source1_entity_id': 'S1-1', 'matched_entity_ids': 'S2-10,S3-10'},
            {'source1_entity_id': 'S1-2', 'matched_entity_ids': ''},
            {'source1_entity_id': 'S1-3', 'matched_entity_ids': 'S2-30'},
            {'source1_entity_id': 'S1-4', 'matched_entity_ids': 'S2-40,S3-40'}
        ])
        y_pred_df = pd.DataFrame([
            {'source1_entity_id': 'S1-1', 'matched_entity_ids': 'S2-10,S3-10'},
            {'source1_entity_id': 'S1-2', 'matched_entity_ids': ''},
            {'source1_entity_id': 'S1-3', 'matched_entity_ids': ''},
            {'source1_entity_id': 'S1-4', 'matched_entity_ids': 'S2-40'}
        ])
        
        expected_macro = (1.0 + 1.0 + 0.0 + (5.0 / 6.0)) / 4.0
        macro_score = score_f05(y_true_df, y_pred_df)
        self.assertAlmostEqual(macro_score, expected_macro, places=6)


if __name__ == '__main__':
    unittest.main()
