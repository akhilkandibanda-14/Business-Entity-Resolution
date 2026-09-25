"""
Competition evaluation metric module for Business Entity Resolution.
Calculates macro-averaged F0.5 score across all Source 1 entities.
"""

from typing import Dict, Set, Union, List, Tuple
import pandas as pd
import numpy as np


def calculate_single_entity_f05(
    true_ids: Set[str],
    pred_ids: Set[str]
) -> float:
    """
    Calculate F0.5 score for a single Source 1 entity.
    
    Formula:
        F0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
        
    Special cases:
        - If both true_ids and pred_ids are empty -> 1.0 (correctly predicted no matches)
        - If true_ids is empty and pred_ids is non-empty -> 0.0 (false positive prediction for a singleton)
        - If true_ids is non-empty and pred_ids is empty -> 0.0 (false negative prediction)
        - If precision + recall == 0 -> 0.0
    """
    # Normalize inputs to sets
    if not isinstance(true_ids, set):
        true_ids = set(true_ids) if true_ids else set()
    if not isinstance(pred_ids, set):
        pred_ids = set(pred_ids) if pred_ids else set()
        
    len_true = len(true_ids)
    len_pred = len(pred_ids)
    
    # Case 1: Both empty (true singleton correctly predicted as no match)
    if len_true == 0 and len_pred == 0:
        return 1.0
        
    # Case 2: One is empty while the other is non-empty
    if len_true == 0 or len_pred == 0:
        return 0.0
        
    # Case 3: Both non-empty
    tp = len(true_ids & pred_ids)
    if tp == 0:
        return 0.0
        
    precision = tp / len_pred
    recall = tp / len_true
    
    denom = 0.25 * precision + recall
    if denom == 0.0:
        return 0.0
        
    f05 = (1.25 * precision * recall) / denom
    return float(f05)


def score_f05(
    y_true: Union[Dict[str, Set[str]], pd.DataFrame, List[Tuple[str, Set[str]]]],
    y_pred: Union[Dict[str, Set[str]], pd.DataFrame, List[Tuple[str, Set[str]]]]
) -> float:
    """
    Calculate macro-averaged F0.5 score across all Source 1 entities.
    
    Args:
        y_true: Dictionary mapping source1_entity_id -> set of matched_entity_ids (or DataFrame with columns ['source1_entity_id', 'matched_entity_ids'])
        y_pred: Dictionary mapping source1_entity_id -> set of matched_entity_ids (or DataFrame with columns ['source1_entity_id', 'matched_entity_ids'])
        
    Returns:
        float: Macro-averaged F0.5 score
    """
    # Convert DataFrames to dict if necessary
    if isinstance(y_true, pd.DataFrame):
        dict_true = {}
        for _, row in y_true.iterrows():
            s1 = row['source1_entity_id']
            m = row['matched_entity_ids']
            if pd.isna(m) or not m:
                dict_true[s1] = set()
            elif isinstance(m, str):
                dict_true[s1] = set(m.split(',')) if m.strip() else set()
            elif isinstance(m, (set, list, tuple)):
                dict_true[s1] = set(m)
            else:
                dict_true[s1] = set()
        y_true = dict_true

    if isinstance(y_pred, pd.DataFrame):
        dict_pred = {}
        for _, row in y_pred.iterrows():
            s1 = row['source1_entity_id']
            m = row['matched_entity_ids']
            if pd.isna(m) or not m:
                dict_pred[s1] = set()
            elif isinstance(m, str):
                dict_pred[s1] = set(m.split(',')) if m.strip() else set()
            elif isinstance(m, (set, list, tuple)):
                dict_pred[s1] = set(m)
            else:
                dict_pred[s1] = set()
        y_pred = dict_pred

    # Ensure y_true is dict mapping s1_id -> set
    if isinstance(y_true, list):
        y_true = dict(y_true)
    if isinstance(y_pred, list):
        y_pred = dict(y_pred)

    all_s1_ids = set(y_true.keys())
    if not all_s1_ids:
        return 0.0

    scores = []
    for s1_id in all_s1_ids:
        t_set = y_true.get(s1_id, set())
        p_set = y_pred.get(s1_id, set())
        s_score = calculate_single_entity_f05(t_set, p_set)
        scores.append(s_score)

    return float(np.mean(scores))
