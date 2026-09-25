"""
Token Inverted Index Blocker for Business Entity Resolution.
Memory-efficient dictionary implementation for scalable indexing without OOM.
"""

from typing import Dict, Any, Tuple, Set, List
from collections import defaultdict
import pandas as pd
import numpy as np


class TokenBlocker:
    """
    Blocker 3: Inverted index using significant business name tokens.
    Excludes high-frequency tokens to prevent candidate explosion.
    """
    def __init__(
        self,
        max_df_ratio: float = 0.001,
        max_df_absolute: int = 500,
        max_candidates_per_s1: int = 100,
        use_country_partition: bool = True
    ):
        self.max_df_ratio = max_df_ratio
        self.max_df_absolute = max_df_absolute
        self.max_candidates_per_s1 = max_candidates_per_s1
        self.use_country_partition = use_country_partition
        self.stats: Dict[str, Any] = {}

    def generate_candidates(
        self,
        df_s1: pd.DataFrame,
        df_s2: pd.DataFrame,
        df_s3: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Generates candidates sharing non-frequent significant name tokens.
        
        Returns:
            Tuple of (candidate_dataframe, stats_dictionary)
        """
        s1_tokens = df_s1['business_name_significant_tokens'].fillna('').str.split()
        s2_tokens = df_s2['business_name_significant_tokens'].fillna('').str.split()
        s3_tokens = df_s3['business_name_significant_tokens'].fillna('').str.split()

        total_s23_docs = len(df_s2) + len(df_s3)
        effective_max_df = min(self.max_df_absolute, max(50, int(total_s23_docs * self.max_df_ratio)))

        # 1. Count document frequencies in S23 pool
        df_counts = defaultdict(int)
        for country, tok_list in zip(df_s2['country'], s2_tokens):
            if tok_list:
                c_val = country if self.use_country_partition else 'ALL'
                for t in set(tok_list):
                    df_counts[(c_val, t)] += 1

        for country, tok_list in zip(df_s3['country'], s3_tokens):
            if tok_list:
                c_val = country if self.use_country_partition else 'ALL'
                for t in set(tok_list):
                    df_counts[(c_val, t)] += 1

        vocab_size = len(df_counts)
        valid_tokens = {k for k, v in df_counts.items() if v <= effective_max_df}
        excluded_count = vocab_size - len(valid_tokens)

        posting_sizes = [v for k, v in df_counts.items() if v <= effective_max_df]
        avg_posting = float(np.mean(posting_sizes)) if posting_sizes else 0.0
        max_posting = int(np.max(posting_sizes)) if posting_sizes else 0

        # 2. Populate inverted index for valid tokens
        inv_index = defaultdict(list)
        for eid, country, tok_list in zip(df_s2['entity_id'], df_s2['country'], s2_tokens):
            if tok_list:
                c_val = country if self.use_country_partition else 'ALL'
                for t in set(tok_list):
                    key = (c_val, t)
                    if key in valid_tokens:
                        inv_index[key].append((eid, 'S2'))

        for eid, country, tok_list in zip(df_s3['entity_id'], df_s3['country'], s3_tokens):
            if tok_list:
                c_val = country if self.use_country_partition else 'ALL'
                for t in set(tok_list):
                    key = (c_val, t)
                    if key in valid_tokens:
                        inv_index[key].append((eid, 'S3'))

        # 3. Query S1 entities
        cand_rows = []
        for s1_id, country, tok_list in zip(df_s1['entity_id'], df_s1['country'], s1_tokens):
            if not tok_list:
                continue
            c_val = country if self.use_country_partition else 'ALL'
            cands_for_s1 = set()
            for t in set(tok_list):
                key = (c_val, t)
                if key in inv_index:
                    for cand_id, src in inv_index[key]:
                        cands_for_s1.add((cand_id, src))
                        if len(cands_for_s1) >= self.max_candidates_per_s1:
                            break
                if len(cands_for_s1) >= self.max_candidates_per_s1:
                    break
            for cand_id, src in cands_for_s1:
                cand_rows.append((s1_id, cand_id, src))

        if not cand_rows:
            cand_df = pd.DataFrame(columns=['source1_entity_id', 'candidate_entity_id', 'candidate_source', 'blocker'])
        else:
            cand_df = pd.DataFrame(cand_rows, columns=['source1_entity_id', 'candidate_entity_id', 'candidate_source'])
            cand_df['blocker'] = 'token'

        stats = {
            'vocab_size': vocab_size,
            'excluded_tokens_count': excluded_count,
            'effective_max_df': effective_max_df,
            'avg_posting_list_size': round(avg_posting, 2),
            'max_posting_list_size': max_posting,
            'candidate_count': len(cand_df)
        }
        self.stats = stats

        return cand_df, stats
