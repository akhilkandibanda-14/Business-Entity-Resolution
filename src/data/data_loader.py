"""
Data loading and validation module for Phase 0 setup.
"""

import os
import json
import pandas as pd
from typing import Dict, Any, Tuple, List


DATASET_PATHS = {
    'train_s1': 'dataset/train/train_source1.tsv',
    'train_s2': 'dataset/train/train_source2.tsv',
    'train_s3': 'dataset/train/train_source3.tsv',
    'train_gt': 'dataset/train/train_ground_truth.tsv',
    'test_s1': 'dataset/test/test_source1.tsv',
    'test_s2': 'dataset/test/test_source2.tsv',
    'test_s3': 'dataset/test/test_source3.tsv'
}

EXPECTED_EDA_METRICS = {
    'train_s1': {'rows': 2206821, 'cols': 4, 'columns': ['entity_id', 'business_name', 'business_address', 'country']},
    'train_s2': {'rows': 5034616, 'cols': 4, 'columns': ['entity_id', 'business_name', 'business_address', 'country']},
    'train_s3': {'rows': 5285603, 'cols': 4, 'columns': ['entity_id', 'business_name', 'business_address', 'country']},
    'train_gt': {'rows': 2206821, 'cols': 2, 'columns': ['source1_entity_id', 'matched_entity_ids']},
    'test_s1': {'rows': 1732544, 'cols': 4, 'columns': ['entity_id', 'business_name', 'business_address', 'country']},
    'test_s2': {'rows': 4887273, 'cols': 4, 'columns': ['entity_id', 'business_name', 'business_address', 'country']},
    'test_s3': {'rows': 5082316, 'cols': 4, 'columns': ['entity_id', 'business_name', 'business_address', 'country']}
}


def load_all_datasets(base_path: str = ".") -> Dict[str, pd.DataFrame]:
    """
    Load all 7 TSV files using pd.read_csv(path, sep="\\t").
    Does not modify original dataset files.
    """
    dfs = {}
    for key, rel_path in DATASET_PATHS.items():
        full_path = os.path.join(base_path, rel_path)
        if not os.path.isfile(full_path):
            raise FileNotFoundError(f"Required TSV file not found: {full_path}")
        df = pd.read_csv(full_path, sep="\t")
        dfs[key] = df
    return dfs


def verify_datasets_against_eda(dfs: Dict[str, pd.DataFrame]) -> Tuple[bool, List[str], Dict[str, Dict[str, Any]]]:
    """
    Verify loaded DataFrames against expected EDA metadata.
    Returns (is_matched, list_of_mismatches, observed_summary).
    """
    mismatches = []
    observed_summary = {}

    for key, df in dfs.items():
        n_rows, n_cols = df.shape
        cols = list(df.columns)
        
        expected = EXPECTED_EDA_METRICS.get(key, {})
        exp_rows = expected.get('rows')
        exp_cols = expected.get('cols')
        exp_column_names = expected.get('columns')
        
        observed_summary[key] = {
            'rows': n_rows,
            'cols': n_cols,
            'columns': cols
        }
        
        if n_rows != exp_rows:
            mismatches.append(f"{key}: Row count mismatch. Expected {exp_rows:,}, observed {n_rows:,}.")
        if n_cols != exp_cols:
            mismatches.append(f"{key}: Column count mismatch. Expected {exp_cols}, observed {n_cols}.")
        if cols != exp_column_names:
            mismatches.append(f"{key}: Column names mismatch. Expected {exp_column_names}, observed {cols}.")

    is_matched = len(mismatches) == 0
    return is_matched, mismatches, observed_summary
