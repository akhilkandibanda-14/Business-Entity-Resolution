"""
PIN / Postal Code Blocker module for Business Entity Resolution.
High-performance dictionary inverted index implementation.
"""

from typing import List, Dict
from collections import defaultdict
import pandas as pd
import numpy as np


class PinBlocker:
    """
    Blocker 4: Exact PIN / Postal code match.
    Generates candidates when S1.pin_code == S2/S3.pin_code (and non-empty).
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
        Generates candidate pairs matching exact PIN codes.
        """
        pin_index = defaultdict(list)
        for eid, country, pin in zip(df_s2['entity_id'], df_s2['country'], df_s2['pin_code'].fillna('')):
            pin_str = str(pin).strip()
            if pin_str and pin_str != 'nan':
                c_val = country if self.use_country_partition else 'ALL'
                pin_index[(c_val, pin_str)].append((eid, 'S2'))

        for eid, country, pin in zip(df_s3['entity_id'], df_s3['country'], df_s3['pin_code'].fillna('')):
            pin_str = str(pin).strip()
            if pin_str and pin_str != 'nan':
                c_val = country if self.use_country_partition else 'ALL'
                pin_index[(c_val, pin_str)].append((eid, 'S3'))

        cand_rows = []
        for s1_id, country, pin in zip(df_s1['entity_id'], df_s1['country'], df_s1['pin_code'].fillna('')):
            pin_str = str(pin).strip()
            if not pin_str or pin_str == 'nan':
                continue
            c_val = country if self.use_country_partition else 'ALL'
            matches = pin_index.get((c_val, pin_str), [])
            if self.max_candidates_per_s1 and len(matches) > self.max_candidates_per_s1:
                matches = matches[:self.max_candidates_per_s1]
            for cand_id, src in matches:
                cand_rows.append((s1_id, cand_id, src, 'pin'))

        if not cand_rows:
            return pd.DataFrame(columns=['source1_entity_id', 'candidate_entity_id', 'candidate_source', 'blocker'])

        cand_df = pd.DataFrame(cand_rows, columns=['source1_entity_id', 'candidate_entity_id', 'candidate_source', 'blocker'])
        return cand_df
