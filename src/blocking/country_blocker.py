"""
Country analysis and partitioning helper module for Business Entity Resolution.
"""

from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np


def analyze_country_agreement(
    gt_df: pd.DataFrame,
    df_s1: pd.DataFrame,
    df_s2: pd.DataFrame,
    df_s3: pd.DataFrame
) -> Dict[str, Any]:
    """
    Analyzes country agreement across all true ground truth pairs.
    
    Args:
        gt_df: DataFrame with 'source1_entity_id' and 'matched_entity_ids'
        df_s1: Source 1 DataFrame with 'entity_id' and 'country'
        df_s2: Source 2 DataFrame with 'entity_id' and 'country'
        df_s3: Source 3 DataFrame with 'entity_id' and 'country'
        
    Returns:
        Dict containing country analysis metrics.
    """
    country_map = {}
    country_map.update(dict(zip(df_s1['entity_id'], df_s1['country'])))
    country_map.update(dict(zip(df_s2['entity_id'], df_s2['country'])))
    country_map.update(dict(zip(df_s3['entity_id'], df_s3['country'])))
    
    gt_valid = gt_df.dropna(subset=['matched_entity_ids'])
    
    rows = []
    for s1_id, matched_str in zip(gt_valid['source1_entity_id'], gt_valid['matched_entity_ids']):
        if not matched_str or not isinstance(matched_str, str):
            continue
        for match_id in matched_str.split(','):
            match_id = match_id.strip()
            if match_id:
                rows.append((s1_id, match_id))
                
    pairs_df = pd.DataFrame(rows, columns=['s1_id', 'match_id'])
    total_pairs = len(pairs_df)
    
    if total_pairs == 0:
        return {
            'total_true_pairs': 0,
            'same_country_count': 0,
            'different_country_count': 0,
            'missing_country_cases': 0,
            'agreement_pct': 0.0,
            'is_hard_partition_safe': True
        }
        
    pairs_df['c1'] = pairs_df['s1_id'].map(country_map)
    pairs_df['c2'] = pairs_df['match_id'].map(country_map)
    
    both_non_null = pairs_df[pairs_df['c1'].notna() & pairs_df['c2'].notna()]
    both_count = len(both_non_null)
    
    same_count = (both_non_null['c1'] == both_non_null['c2']).sum() if both_count > 0 else 0
    diff_count = (both_non_null['c1'] != both_non_null['c2']).sum() if both_count > 0 else 0
    missing_count = (pairs_df['c1'].isna() | pairs_df['c2'].isna()).sum()
    
    agreement_pct = (same_count / total_pairs) * 100.0 if total_pairs > 0 else 0.0
    is_safe = (diff_count == 0) and (missing_count == 0)
    
    return {
        'total_true_pairs': total_pairs,
        'same_country_count': int(same_count),
        'different_country_count': int(diff_count),
        'missing_country_cases': int(missing_count),
        'agreement_pct': float(agreement_pct),
        'is_hard_partition_safe': bool(is_safe)
    }


class CountryBlocker:
    """
    Helper for grouping/filtering datasets by country partition dynamically.
    Works with arbitrary open-set country codes (US, IN, FR, etc.).
    """
    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def get_country_groups(
        self,
        df_s1: pd.DataFrame,
        df_s2: pd.DataFrame,
        df_s3: pd.DataFrame
    ) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """
        Splits DataFrames into country-partitioned subsets.
        If country blocking is disabled, returns a single group under key 'ALL'.
        """
        if not self.enabled:
            return {'ALL': (df_s1, df_s2, df_s3)}

        # Find all unique countries across all sources
        all_countries = sorted(list(set(df_s1['country'].dropna().unique()) |
                                    set(df_s2['country'].dropna().unique()) |
                                    set(df_s3['country'].dropna().unique())))

        groups = {}
        for c in all_countries:
            s1_c = df_s1[df_s1['country'] == c]
            s2_c = df_s2[df_s2['country'] == c]
            s3_c = df_s3[df_s3['country'] == c]
            if len(s1_c) > 0 and (len(s2_c) > 0 or len(s3_c) > 0):
                groups[c] = (s1_c, s2_c, s3_c)

        return groups
