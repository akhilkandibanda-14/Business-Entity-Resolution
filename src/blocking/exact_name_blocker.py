"""
Exact Name and Sorted Token Key Blocker modules for Business Entity Resolution.
High-performance dictionary inverted index implementation.
"""

from typing import List, Optional, Tuple, Dict
from collections import defaultdict
import pandas as pd
import numpy as np


class ExactNameBlocker:
    """
    Blocker 1: Generates candidates sharing exact normalized name
    or exact suffix-stripped name.
    """
    def __init__(
        self,
        use_country_partition: bool = True,
        max_candidates_per_s1: int = 100
    ):
        self.use_country_partition = use_country_partition
        self.max_candidates_per_s1 = max_candidates_per_s1

    def generate_candidates(
        self,
        df_s1: pd.DataFrame,
        df_s2: pd.DataFrame,
        df_s3: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generates candidates matching exact normalized business names
        or exact suffix-stripped business names.
        """
        cand_rows = []

        # 1. Index business_name_norm
        norm_index = defaultdict(list)
        for eid, country, norm in zip(df_s2['entity_id'], df_s2['country'], df_s2['business_name_norm'].fillna('')):
            if norm and norm != 'nan':
                c_val = country if self.use_country_partition else 'ALL'
                norm_index[(c_val, norm)].append((eid, 'S2'))

        for eid, country, norm in zip(df_s3['entity_id'], df_s3['country'], df_s3['business_name_norm'].fillna('')):
            if norm and norm != 'nan':
                c_val = country if self.use_country_partition else 'ALL'
                norm_index[(c_val, norm)].append((eid, 'S3'))

        for s1_id, country, norm in zip(df_s1['entity_id'], df_s1['country'], df_s1['business_name_norm'].fillna('')):
            if not norm or norm == 'nan':
                continue
            c_val = country if self.use_country_partition else 'ALL'
            matches = norm_index.get((c_val, norm), [])
            if self.max_candidates_per_s1 and len(matches) > self.max_candidates_per_s1:
                matches = matches[:self.max_candidates_per_s1]
            for cand_id, src in matches:
                cand_rows.append((s1_id, cand_id, src, 'exact_name_norm'))

        # 2. Index business_name_no_suffix
        nosuff_index = defaultdict(list)
        for eid, country, nosuff in zip(df_s2['entity_id'], df_s2['country'], df_s2['business_name_no_suffix'].fillna('')):
            if nosuff and nosuff != 'nan':
                c_val = country if self.use_country_partition else 'ALL'
                nosuff_index[(c_val, nosuff)].append((eid, 'S2'))

        for eid, country, nosuff in zip(df_s3['entity_id'], df_s3['country'], df_s3['business_name_no_suffix'].fillna('')):
            if nosuff and nosuff != 'nan':
                c_val = country if self.use_country_partition else 'ALL'
                nosuff_index[(c_val, nosuff)].append((eid, 'S3'))

        for s1_id, country, nosuff in zip(df_s1['entity_id'], df_s1['country'], df_s1['business_name_no_suffix'].fillna('')):
            if not nosuff or nosuff == 'nan':
                continue
            c_val = country if self.use_country_partition else 'ALL'
            matches = nosuff_index.get((c_val, nosuff), [])
            if self.max_candidates_per_s1 and len(matches) > self.max_candidates_per_s1:
                matches = matches[:self.max_candidates_per_s1]
            for cand_id, src in matches:
                cand_rows.append((s1_id, cand_id, src, 'exact_name_no_suffix'))

        if not cand_rows:
            return pd.DataFrame(columns=['source1_entity_id', 'candidate_entity_id', 'candidate_source', 'blocker'])

        cand_df = pd.DataFrame(cand_rows, columns=['source1_entity_id', 'candidate_entity_id', 'candidate_source', 'blocker'])
        return cand_df


class SortedTokenBlocker:
    """
    Blocker 2: Generates candidates sharing sorted canonical significant tokens key.
    E.g. "Global Technologies India Pvt Ltd" -> "global|india|technologies"
    """
    def __init__(
        self,
        use_country_partition: bool = True,
        max_candidates_per_s1: int = 100
    ):
        self.use_country_partition = use_country_partition
        self.max_candidates_per_s1 = max_candidates_per_s1

    @staticmethod
    def _create_sorted_token_key(tokens_val) -> str:
        if not tokens_val or pd.isna(tokens_val):
            return ""
        if isinstance(tokens_val, str):
            tok_list = [t for t in tokens_val.split() if t]
        else:
            tok_list = list(tokens_val)
        if not tok_list:
            return ""
        unique_sorted = sorted(list(set(tok_list)))
        return "|".join(unique_sorted)

    def generate_candidates(
        self,
        df_s1: pd.DataFrame,
        df_s2: pd.DataFrame,
        df_s3: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generates candidates matching canonical sorted token keys.
        """
        s1_keys = df_s1['business_name_significant_tokens'].fillna('').apply(self._create_sorted_token_key)
        s2_keys = df_s2['business_name_significant_tokens'].fillna('').apply(self._create_sorted_token_key)
        s3_keys = df_s3['business_name_significant_tokens'].fillna('').apply(self._create_sorted_token_key)

        key_index = defaultdict(list)
        for eid, country, key_val in zip(df_s2['entity_id'], df_s2['country'], s2_keys):
            if key_val:
                c_val = country if self.use_country_partition else 'ALL'
                key_index[(c_val, key_val)].append((eid, 'S2'))

        for eid, country, key_val in zip(df_s3['entity_id'], df_s3['country'], s3_keys):
            if key_val:
                c_val = country if self.use_country_partition else 'ALL'
                key_index[(c_val, key_val)].append((eid, 'S3'))

        cand_rows = []
        for s1_id, country, key_val in zip(df_s1['entity_id'], df_s1['country'], s1_keys):
            if not key_val:
                continue
            c_val = country if self.use_country_partition else 'ALL'
            matches = key_index.get((c_val, key_val), [])
            if self.max_candidates_per_s1 and len(matches) > self.max_candidates_per_s1:
                matches = matches[:self.max_candidates_per_s1]
            for cand_id, src in matches:
                cand_rows.append((s1_id, cand_id, src, 'sorted_token'))

        if not cand_rows:
            return pd.DataFrame(columns=['source1_entity_id', 'candidate_entity_id', 'candidate_source', 'blocker'])

        cand_df = pd.DataFrame(cand_rows, columns=['source1_entity_id', 'candidate_entity_id', 'candidate_source', 'blocker'])
        return cand_df
