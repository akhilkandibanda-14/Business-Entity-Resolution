"""
Evaluation and blocking metrics module for Business Entity Resolution Phase 2.
Vectorized Pandas implementation for ultra-fast recall and volume metrics.
"""

from typing import Dict, Any, List, Set, Tuple, Optional
import pandas as pd
import numpy as np


def calculate_blocking_metrics(
    candidate_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    df_s1: pd.DataFrame,
    df_s2: pd.DataFrame,
    df_s3: pd.DataFrame,
    use_country_partition: bool = True
) -> Dict[str, Any]:
    """
    Computes pair-level recall, entity-level recall, volume metrics, and reduction ratios.
    Vectorized using Pandas joins for maximum performance.
    """
    # 1. Explode ground truth into (source1_entity_id, match_id)
    gt_valid = gt_df.dropna(subset=['matched_entity_ids'])

    gt_s1 = []
    gt_m = []
    for s1_id, matched_str in zip(gt_valid['source1_entity_id'], gt_valid['matched_entity_ids']):
        if matched_str and isinstance(matched_str, str):
            for m in matched_str.split(','):
                m_clean = m.strip()
                if m_clean:
                    gt_s1.append(s1_id)
                    gt_m.append(m_clean)

    df_gt_pairs = pd.DataFrame({'source1_entity_id': gt_s1, 'candidate_entity_id': gt_m})
    total_true_pairs = len(df_gt_pairs)
    total_s1_with_matches = df_gt_pairs['source1_entity_id'].nunique()

    total_candidate_pairs = len(candidate_df)

    if total_candidate_pairs == 0 or total_true_pairs == 0:
        return {
            'total_true_pairs': total_true_pairs,
            'retained_pair_count': 0,
            'blocking_recall_pct': 0.0,
            'total_s1_with_matches': total_s1_with_matches,
            'complete_entity_recall_pct': 0.0,
            'at_least_one_entity_recall_pct': 0.0,
            'total_candidates': 0,
            'avg_candidates_per_s1': 0.0,
            'median_candidates_per_s1': 0.0,
            'p90_candidates_per_s1': 0.0,
            'p95_candidates_per_s1': 0.0,
            'p99_candidates_per_s1': 0.0,
            'max_candidates_per_s1': 0,
            'avg_candidates_no_match_s1': 0.0,
            'raw_possible_pairs': len(df_s1) * (len(df_s2) + len(df_s3)),
            'raw_reduction_ratio_pct': 100.0,
            'country_possible_pairs': len(df_s1) * (len(df_s2) + len(df_s3)),
            'country_reduction_ratio_pct': 100.0
        }

    # Deduplicate candidate pairs
    cand_pairs_unique = candidate_df[['source1_entity_id', 'candidate_entity_id']].drop_duplicates()
    total_candidate_pairs = len(cand_pairs_unique)

    # 2. Vectorized Pair-Level Recall via inner merge
    retained_df = pd.merge(
        df_gt_pairs,
        cand_pairs_unique,
        on=['source1_entity_id', 'candidate_entity_id'],
        how='inner'
    )
    retained_pair_count = len(retained_df)
    pair_blocking_recall = (retained_pair_count / total_true_pairs) * 100.0 if total_true_pairs > 0 else 0.0

    # 3. Complete Entity Recall & At Least One Entity Recall
    gt_target_counts = df_gt_pairs.groupby('source1_entity_id').size().rename('total_gt')
    retained_counts = retained_df.groupby('source1_entity_id').size().rename('retained_gt')

    entity_eval_df = pd.concat([gt_target_counts, retained_counts], axis=1).fillna(0)
    
    complete_count = (entity_eval_df['total_gt'] == entity_eval_df['retained_gt']).sum()
    at_least_one_count = (entity_eval_df['retained_gt'] > 0).sum()

    complete_entity_recall_pct = (complete_count / total_s1_with_matches) * 100.0 if total_s1_with_matches > 0 else 0.0
    at_least_one_entity_recall_pct = (at_least_one_count / total_s1_with_matches) * 100.0 if total_s1_with_matches > 0 else 0.0

    # 4. Candidate Volume Percentiles per S1 Entity
    cand_per_s1 = cand_pairs_unique.groupby('source1_entity_id').size()
    s1_all_series = pd.Series(0, index=df_s1['entity_id'])
    s1_all_series.update(cand_per_s1)

    cand_counts_array = s1_all_series.values
    mean_cand = float(np.mean(cand_counts_array))
    median_cand = float(np.median(cand_counts_array))
    p90_cand = float(np.percentile(cand_counts_array, 90))
    p95_cand = float(np.percentile(cand_counts_array, 95))
    p99_cand = float(np.percentile(cand_counts_array, 99))
    max_cand = int(np.max(cand_counts_array))

    # Candidate volume for true no-match entities
    gt_s1_set = set(gt_target_counts.index)
    no_match_mask = ~s1_all_series.index.isin(gt_s1_set)
    no_match_counts = s1_all_series[no_match_mask].values
    avg_cand_no_match = float(np.mean(no_match_counts)) if len(no_match_counts) > 0 else 0.0

    # 5. Reduction Ratios
    n1 = len(df_s1)
    n2 = len(df_s2)
    n3 = len(df_s3)
    raw_possible_pairs = n1 * (n2 + n3)
    raw_reduction_ratio = (1.0 - (total_candidate_pairs / raw_possible_pairs)) * 100.0 if raw_possible_pairs > 0 else 0.0

    country_possible_pairs = raw_possible_pairs
    if use_country_partition and 'country' in df_s1.columns:
        s1_counts = df_s1['country'].value_counts().to_dict()
        s2_counts = df_s2['country'].value_counts().to_dict()
        s3_counts = df_s3['country'].value_counts().to_dict()

        c_possible = 0
        for c, c1_n in s1_counts.items():
            c2_n = s2_counts.get(c, 0)
            c3_n = s3_counts.get(c, 0)
            c_possible += c1_n * (c2_n + c3_n)

        country_possible_pairs = c_possible
        country_reduction_ratio = (1.0 - (total_candidate_pairs / country_possible_pairs)) * 100.0 if country_possible_pairs > 0 else 0.0
    else:
        country_reduction_ratio = raw_reduction_ratio

    return {
        'total_true_pairs': total_true_pairs,
        'retained_pair_count': retained_pair_count,
        'blocking_recall_pct': round(pair_blocking_recall, 4),
        'total_s1_with_matches': total_s1_with_matches,
        'complete_entity_recall_pct': round(complete_entity_recall_pct, 4),
        'at_least_one_entity_recall_pct': round(at_least_one_entity_recall_pct, 4),
        'total_candidates': total_candidate_pairs,
        'avg_candidates_per_s1': round(mean_cand, 2),
        'median_candidates_per_s1': round(median_cand, 2),
        'p90_candidates_per_s1': round(p90_cand, 2),
        'p95_candidates_per_s1': round(p95_cand, 2),
        'p99_candidates_per_s1': round(p99_cand, 2),
        'max_candidates_per_s1': max_cand,
        'avg_candidates_no_match_s1': round(avg_cand_no_match, 2),
        'raw_possible_pairs': raw_possible_pairs,
        'raw_reduction_ratio_pct': round(raw_reduction_ratio, 6),
        'country_possible_pairs': country_possible_pairs,
        'country_reduction_ratio_pct': round(country_reduction_ratio, 6)
    }
