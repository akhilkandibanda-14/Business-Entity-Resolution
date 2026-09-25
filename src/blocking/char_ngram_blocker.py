"""
Character N-Gram Inverted Index Blocker for Business Entity Resolution.
Memory-efficient dictionary implementation for scalable indexing without OOM.
"""

from typing import Dict, Any, Tuple, Set, List
from collections import defaultdict, Counter
import pandas as pd
import numpy as np


class CharNgramBlocker:
    """
    Blocker 5: Inverted index using character trigrams.
    Excludes high-frequency trigrams and requires a minimum shared trigram count.
    """
    def __init__(
        self,
        min_shared_trigrams: int = 3,
        max_df_ratio: float = 0.005,
        max_df_absolute: int = 1000,
        max_candidates_per_s1: int = 100,
        use_country_partition: bool = True
    ):
        self.min_shared_trigrams = min_shared_trigrams
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
        Generates candidate pairs meeting minimum character trigram overlap threshold.
        """
        s1_tri = df_s1['business_name_char_trigrams'].fillna('').str.split()
        s2_tri = df_s2['business_name_char_trigrams'].fillna('').str.split()
        s3_tri = df_s3['business_name_char_trigrams'].fillna('').str.split()

        total_s23_docs = len(df_s2) + len(df_s3)
        effective_max_df = min(self.max_df_absolute, max(100, int(total_s23_docs * self.max_df_ratio)))

        # 1. Count trigram document frequencies
        df_counts = defaultdict(int)
        for country, tri_list in zip(df_s2['country'], s2_tri):
            if tri_list:
                c_val = country if self.use_country_partition else 'ALL'
                for t in set(tri_list):
                    df_counts[(c_val, t)] += 1

        for country, tri_list in zip(df_s3['country'], s3_tri):
            if tri_list:
                c_val = country if self.use_country_partition else 'ALL'
                for t in set(tri_list):
                    df_counts[(c_val, t)] += 1

        vocab_size = len(df_counts)
        valid_trigrams = {k for k, v in df_counts.items() if v <= effective_max_df}
        excluded_count = vocab_size - len(valid_trigrams)

        # 2. Populate inverted index for valid trigrams
        inv_index = defaultdict(list)
        for eid, country, tri_list in zip(df_s2['entity_id'], df_s2['country'], s2_tri):
            if tri_list:
                c_val = country if self.use_country_partition else 'ALL'
                for t in set(tri_list):
                    key = (c_val, t)
                    if key in valid_trigrams:
                        inv_index[key].append((eid, 'S2'))

        for eid, country, tri_list in zip(df_s3['entity_id'], df_s3['country'], s3_tri):
            if tri_list:
                c_val = country if self.use_country_partition else 'ALL'
                for t in set(tri_list):
                    key = (c_val, t)
                    if key in valid_trigrams:
                        inv_index[key].append((eid, 'S3'))

        # 3. Query S1 entities and count shared trigrams
        cand_rows = []
        for s1_id, country, tri_list in zip(df_s1['entity_id'], df_s1['country'], s1_tri):
            if not tri_list:
                continue
            c_val = country if self.use_country_partition else 'ALL'
            overlap_counts = Counter()
            for t in set(tri_list):
                key = (c_val, t)
                if key in inv_index:
                    for item in inv_index[key]:
                        overlap_counts[item] += 1

            matched_items = [item for item, count in overlap_counts.items() if count >= self.min_shared_trigrams]
            if self.max_candidates_per_s1 and len(matched_items) > self.max_candidates_per_s1:
                # Top candidates by overlap count
                matched_items = sorted(matched_items, key=lambda item: overlap_counts[item], reverse=True)[:self.max_candidates_per_s1]

            for cand_id, src in matched_items:
                cand_rows.append((s1_id, cand_id, src))

        if not cand_rows:
            cand_df = pd.DataFrame(columns=['source1_entity_id', 'candidate_entity_id', 'candidate_source', 'blocker'])
        else:
            cand_df = pd.DataFrame(cand_rows, columns=['source1_entity_id', 'candidate_entity_id', 'candidate_source'])
            cand_df['blocker'] = 'char_ngram'

        stats = {
            'vocab_size': vocab_size,
            'excluded_trigrams_count': excluded_count,
            'effective_max_df': effective_max_df,
            'min_shared_trigrams': self.min_shared_trigrams,
            'candidate_count': len(cand_df)
        }
        self.stats = stats

        return cand_df, stats
