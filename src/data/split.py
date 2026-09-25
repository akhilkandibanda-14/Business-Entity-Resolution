"""
Reproducible validation splitting module for Phase 0 setup.
Splits dataset at the Source 1 entity level.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any


def categorize_match_type(matched_str: str) -> str:
    """Categorize ground truth match into one of 4 mutually exclusive types."""
    if pd.isna(matched_str) or not matched_str or not str(matched_str).strip():
        return 'no_match'
    matches = str(matched_str).split(',')
    has_s2 = any(m.startswith('S2-') for m in matches)
    has_s3 = any(m.startswith('S3-') for m in matches)
    
    if has_s2 and has_s3:
        return 'both'
    elif has_s2:
        return 's2_only'
    elif has_s3:
        return 's3_only'
    return 'no_match'


def create_source1_validation_split(
    gt_df: pd.DataFrame,
    test_size: float = 0.10,
    random_seed: int = 42,
    output_dir: str = "artifacts/validation"
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Creates a reproducible holdout split based on Source 1 entities.
    
    Args:
        gt_df: DataFrame containing 'source1_entity_id' and 'matched_entity_ids'
        test_size: Ratio of validation set (default 0.10 = 10%)
        random_seed: Fixed seed for reproducibility (default 42)
        output_dir: Directory where validation artifacts will be saved
        
    Returns:
        (dev_gt_df, val_gt_df, summary_stats)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Categorize match types for stratified splitting
    gt_copy = gt_df.copy()
    gt_copy['match_category'] = gt_copy['matched_entity_ids'].apply(categorize_match_type)
    
    # 2. Stratified train/validation split at Source 1 entity level
    dev_gt, val_gt = train_test_split(
        gt_copy,
        test_size=test_size,
        random_state=random_seed,
        stratify=gt_copy['match_category']
    )
    
    dev_s1_ids = dev_gt['source1_entity_id'].values
    val_s1_ids = val_gt['source1_entity_id'].values
    
    # Verify no data leakage
    overlap = set(dev_s1_ids) & set(val_s1_ids)
    if len(overlap) > 0:
        raise ValueError(f"CRITICAL ERROR: Data leakage detected! {len(overlap)} S1 IDs overlap between dev and val splits.")
        
    # 3. Save ID files
    dev_id_file = os.path.join(output_dir, "dev_source1_ids.txt")
    val_id_file = os.path.join(output_dir, "validation_source1_ids.txt")
    
    with open(dev_id_file, "w", encoding="utf-8") as f:
        for s1_id in dev_s1_ids:
            f.write(f"{s1_id}\n")
            
    with open(val_id_file, "w", encoding="utf-8") as f:
        for s1_id in val_s1_ids:
            f.write(f"{s1_id}\n")
            
    # 4. Save ground-truth TSV files for dev and validation sets
    dev_gt_tsv = os.path.join(output_dir, "dev_ground_truth.tsv")
    val_gt_tsv = os.path.join(output_dir, "validation_ground_truth.tsv")
    
    dev_gt[['source1_entity_id', 'matched_entity_ids']].to_csv(dev_gt_tsv, sep="\t", index=False)
    val_gt[['source1_entity_id', 'matched_entity_ids']].to_csv(val_gt_tsv, sep="\t", index=False)
    
    # 5. Compute validation ground-truth statistics
    val_total = len(val_gt)
    val_no_matches = int((val_gt['match_category'] == 'no_match').sum())
    
    # Calculate match count distribution for val_gt
    def get_num_matches(val):
        if pd.isna(val) or not val or not str(val).strip():
            return 0
        return len(str(val).split(','))
        
    val_match_counts = val_gt['matched_entity_ids'].apply(get_num_matches)
    val_one_match = int((val_match_counts == 1).sum())
    val_multi_match = int((val_match_counts > 1).sum())
    
    stats = {
        'total_s1_entities': len(gt_copy),
        'dev_size': len(dev_gt),
        'validation_size': len(val_gt),
        'test_size_ratio': test_size,
        'random_seed': random_seed,
        'leakage_detected': False,
        'val_zero_matches': val_no_matches,
        'val_zero_matches_pct': round(val_no_matches / val_total * 100, 4),
        'val_one_match': val_one_match,
        'val_one_match_pct': round(val_one_match / val_total * 100, 4),
        'val_multi_match': val_multi_match,
        'val_multi_match_pct': round(val_multi_match / val_total * 100, 4),
        'val_s2_only': int((val_gt['match_category'] == 's2_only').sum()),
        'val_s3_only': int((val_gt['match_category'] == 's3_only').sum()),
        'val_both': int((val_gt['match_category'] == 'both').sum())
    }
    
    return dev_gt, val_gt, stats
