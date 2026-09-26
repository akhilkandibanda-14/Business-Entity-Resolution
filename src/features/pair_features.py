"""
Source, blocker provenance, and composite pairwise feature extraction module for Phase 3.
"""

from typing import Dict, Any, Union
import pandas as pd
import numpy as np


def compute_pair_features_dict(
    cand_source: str,
    blockers_str: str,
    name_feats: Dict[str, Any],
    addr_feats: Dict[str, Any]
) -> Dict[str, Any]:
    """Computes source indicators, blocker provenance, and composite similarity features."""
    src_str = str(cand_source or '').strip().upper()
    blockers_val = str(blockers_str or '').strip()

    candidate_is_s2 = 1 if 'S2' in src_str else 0
    candidate_is_s3 = 1 if 'S3' in src_str else 0
    source_pair = 1 if candidate_is_s2 else 2 # 1 for S1-S2, 2 for S1-S3

    # Blocker provenance indicators
    blocked_exact_name = 1 if ('exact' in blockers_val or 'exact_name_norm' in blockers_val or 'exact_name_no_suffix' in blockers_val) else 0
    blocked_sorted_token = 1 if 'sorted_token' in blockers_val else 0
    blocked_token = 1 if 'token' in blockers_val and 'sorted_token' not in blockers_val or 'token' in blockers_val.split('|') else 0
    blocked_pin = 1 if 'pin' in blockers_val else 0
    blocked_char_ngram = 1 if 'char_ngram' in blockers_val else 0

    if blockers_val:
        num_blockers = len(set(b.strip() for b in blockers_val.split('|') if b.strip()))
    else:
        num_blockers = 1

    # Composite similarity features
    name_sims = [
        name_feats.get('name_token_jaccard', 0.0),
        name_feats.get('name_char_trigram_jaccard', 0.0),
        name_feats.get('name_levenshtein', 0.0),
        name_feats.get('name_jaro_winkler', 0.0)
    ]
    name_similarity_mean = float(np.mean(name_sims))

    addr_sims = [
        addr_feats.get('address_token_jaccard', 0.0),
        addr_feats.get('address_char_trigram_jaccard', 0.0),
        addr_feats.get('address_levenshtein', 0.0)
    ]
    address_similarity_mean = float(np.mean(addr_sims))

    overall_text_similarity = float(0.6 * name_similarity_mean + 0.4 * address_similarity_mean)

    return {
        'candidate_is_s2': candidate_is_s2,
        'candidate_is_s3': candidate_is_s3,
        'source_pair': source_pair,
        'blocked_exact_name': blocked_exact_name,
        'blocked_sorted_token': blocked_sorted_token,
        'blocked_token': blocked_token,
        'blocked_pin': blocked_pin,
        'blocked_char_ngram': blocked_char_ngram,
        'num_blockers': num_blockers,
        'name_similarity_mean': round(name_similarity_mean, 6),
        'address_similarity_mean': round(address_similarity_mean, 6),
        'overall_text_similarity': round(overall_text_similarity, 6)
    }
