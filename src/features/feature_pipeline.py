"""
Feature engineering pipeline module for Business Entity Resolution Phase 3.
Combines candidate pairs with normalized data, computes pairwise features,
and validates feature quality.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

from src.features.name_features import compute_name_features_dict
from src.features.address_features import compute_address_features_dict
from src.features.pair_features import compute_pair_features_dict


class FeaturePipeline:
    """
    Computes pairwise similarity features for candidate pairs in batches.
    """
    def __init__(self, batch_size: int = 250000):
        self.batch_size = batch_size

    @staticmethod
    def build_entity_dicts(
        df_s1: pd.DataFrame,
        df_s2: pd.DataFrame,
        df_s3: pd.DataFrame
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        """Creates entity attribute lookup dictionaries for O(1) row access."""
        s1_dict = df_s1.set_index('entity_id').to_dict('index')
        s2_dict = df_s2.set_index('entity_id').to_dict('index')
        s3_dict = df_s3.set_index('entity_id').to_dict('index')
        return s1_dict, s2_dict, s3_dict

    def extract_features_for_candidates(
        self,
        candidate_df: pd.DataFrame,
        df_s1: pd.DataFrame,
        df_s2: pd.DataFrame,
        df_s3: pd.DataFrame,
        gt_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Extracts pairwise features for all candidate pairs in candidate_df.
        
        Args:
            candidate_df: DataFrame with 'source1_entity_id', 'candidate_entity_id', 'candidate_source', 'blockers'
            df_s1: Source 1 normalized DataFrame
            df_s2: Source 2 normalized DataFrame
            df_s3: Source 3 normalized DataFrame
            gt_df: Optional ground truth DataFrame to generate 'label' column
            
        Returns:
            DataFrame containing candidate identifiers, feature columns, and optional label.
        """
        if len(candidate_df) == 0:
            return pd.DataFrame()

        # Build ground truth lookup set if GT provided
        gt_set = set()
        if gt_df is not None and len(gt_df) > 0:
            gt_valid = gt_df.dropna(subset=['matched_entity_ids'])
            for s1_id, matched_str in zip(gt_valid['source1_entity_id'], gt_valid['matched_entity_ids']):
                if matched_str and isinstance(matched_str, str):
                    for m in matched_str.split(','):
                        m_clean = m.strip()
                        if m_clean:
                            gt_set.add((s1_id, m_clean))

        s1_dict, s2_dict, s3_dict = self.build_entity_dicts(df_s1, df_s2, df_s3)

        rows = []

        s1_ids = candidate_df['source1_entity_id'].values
        cand_ids = candidate_df['candidate_entity_id'].values
        srcs = candidate_df['candidate_source'].values
        blockers_arr = candidate_df['blockers'].values if 'blockers' in candidate_df.columns else [""] * len(candidate_df)

        for s1_id, cand_id, src_val, b_val in zip(s1_ids, cand_ids, srcs, blockers_arr):
            s1_rec = s1_dict.get(s1_id, {})
            
            if 'S2' in str(src_val).upper():
                cand_rec = s2_dict.get(cand_id, {})
            else:
                cand_rec = s3_dict.get(cand_id, {})

            name_feats = compute_name_features_dict(s1_rec, cand_rec)
            addr_feats = compute_address_features_dict(s1_rec, cand_rec)
            pair_feats = compute_pair_features_dict(src_val, b_val, name_feats, addr_feats)

            row = {
                'source1_entity_id': s1_id,
                'candidate_entity_id': cand_id,
                'candidate_source': src_val
            }
            row.update(name_feats)
            row.update(addr_feats)
            row.update(pair_feats)

            if gt_df is not None:
                row['label'] = 1 if (s1_id, cand_id) in gt_set else 0

            rows.append(row)

        feature_df = pd.DataFrame(rows)

        # Validate feature quality (check NaN, inf)
        self.validate_feature_quality(feature_df, has_label=(gt_df is not None))

        return feature_df

    @staticmethod
    def validate_feature_quality(df: pd.DataFrame, has_label: bool = True):
        """
        Checks feature dataframe for NaN, infinite values, and unexpected ranges.
        Raises ValueError if invalid data quality detected.
        """
        if len(df) == 0:
            return

        # Check NaN
        nan_cols = df.isna().sum()
        has_nans = nan_cols[nan_cols > 0]
        if len(has_nans) > 0:
            raise ValueError(f"Feature engineering dataset contains NaN values in columns: {has_nans.to_dict()}")

        # Check infinity in numeric columns
        num_cols = df.select_dtypes(include=[np.number]).columns
        inf_counts = np.isinf(df[num_cols].values).sum()
        if inf_counts > 0:
            raise ValueError(f"Feature engineering dataset contains {inf_counts} infinite (+inf/-inf) values!")

        # Check binary/range features
        if has_label and 'label' in df.columns:
            invalid_labels = df[~df['label'].isin([0, 1])]
            if len(invalid_labels) > 0:
                raise ValueError("Target label contains invalid values outside {0, 1}!")
