"""
Candidate Generator pipeline combining independent blocking strategies into a unified,
deduplicated candidate set with blocker provenance.
Fast dictionary aggregation implementation.
"""

from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import pandas as pd
import numpy as np

from src.blocking.country_blocker import CountryBlocker
from src.blocking.exact_name_blocker import ExactNameBlocker, SortedTokenBlocker
from src.blocking.token_blocker import TokenBlocker
from src.blocking.pin_blocker import PinBlocker
from src.blocking.char_ngram_blocker import CharNgramBlocker


class CandidateGenerator:
    """
    Orchestrates candidate generation across independent blockers,
    merges candidates via UNION, deduplicates, and tracks blocker provenance.
    """
    def __init__(
        self,
        use_country_partition: bool = True,
        max_token_df_ratio: float = 0.001,
        max_token_df_absolute: int = 500,
        min_shared_trigrams: int = 3,
        max_trigram_df_ratio: float = 0.005,
        max_trigram_df_absolute: int = 1000
    ):
        self.use_country_partition = use_country_partition
        self.country_blocker = CountryBlocker(enabled=use_country_partition)
        self.exact_blocker = ExactNameBlocker(use_country_partition=use_country_partition)
        self.sorted_token_blocker = SortedTokenBlocker(use_country_partition=use_country_partition)
        self.token_blocker = TokenBlocker(
            max_df_ratio=max_token_df_ratio,
            max_df_absolute=max_token_df_absolute,
            use_country_partition=use_country_partition
        )
        self.pin_blocker = PinBlocker(use_country_partition=use_country_partition)
        self.char_ngram_blocker = CharNgramBlocker(
            min_shared_trigrams=min_shared_trigrams,
            max_df_ratio=max_trigram_df_ratio,
            max_df_absolute=max_trigram_df_absolute,
            use_country_partition=use_country_partition
        )

    def generate_all_candidates(
        self,
        df_s1: pd.DataFrame,
        df_s2: pd.DataFrame,
        df_s3: pd.DataFrame,
        enabled_blockers: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Runs enabled blockers, combines candidates via UNION, deduplicates,
        and aggregates blocker provenance.
        """
        if enabled_blockers is None:
            enabled_blockers = ['exact', 'sorted_token', 'token', 'pin', 'char_ngram']

        blocker_outputs = {}
        blocker_stats = {}

        # 1. Exact name blocker
        if 'exact' in enabled_blockers:
            df_exact = self.exact_blocker.generate_candidates(df_s1, df_s2, df_s3)
            blocker_outputs['exact'] = df_exact
            blocker_stats['exact'] = {'candidate_count': len(df_exact)}

        # 2. Sorted token blocker
        if 'sorted_token' in enabled_blockers:
            df_sorted = self.sorted_token_blocker.generate_candidates(df_s1, df_s2, df_s3)
            blocker_outputs['sorted_token'] = df_sorted
            blocker_stats['sorted_token'] = {'candidate_count': len(df_sorted)}

        # 3. Token inverted index blocker
        if 'token' in enabled_blockers:
            df_tok, tok_stat = self.token_blocker.generate_candidates(df_s1, df_s2, df_s3)
            blocker_outputs['token'] = df_tok
            blocker_stats['token'] = tok_stat

        # 4. PIN blocker
        if 'pin' in enabled_blockers:
            df_pin = self.pin_blocker.generate_candidates(df_s1, df_s2, df_s3)
            blocker_outputs['pin'] = df_pin
            blocker_stats['pin'] = {'candidate_count': len(df_pin)}

        # 5. Char n-gram blocker
        if 'char_ngram' in enabled_blockers:
            df_ngram, ngram_stat = self.char_ngram_blocker.generate_candidates(df_s1, df_s2, df_s3)
            blocker_outputs['char_ngram'] = df_ngram
            blocker_stats['char_ngram'] = ngram_stat

        # Aggregate candidates using fast dictionary lookup
        cand_dict = defaultdict(lambda: {'src': '', 'blockers': []})

        for df_b in blocker_outputs.values():
            if len(df_b) > 0:
                for s1, cand, src, b in zip(df_b['source1_entity_id'], df_b['candidate_entity_id'], df_b['candidate_source'], df_b['blocker']):
                    item = cand_dict[(s1, cand)]
                    item['src'] = src
                    item['blockers'].append(b)

        if not cand_dict:
            empty_df = pd.DataFrame(columns=['source1_entity_id', 'candidate_entity_id', 'candidate_source', 'blockers'])
            return empty_df, blocker_stats

        rows = [(k[0], k[1], v['src'], '|'.join(sorted(list(set(v['blockers']))))) for k, v in cand_dict.items()]
        unified_df = pd.DataFrame(rows, columns=['source1_entity_id', 'candidate_entity_id', 'candidate_source', 'blockers'])

        return unified_df, blocker_stats
