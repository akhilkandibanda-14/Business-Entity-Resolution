"""
Business name pairwise feature extraction module for Phase 3.
"""

from typing import Dict, Any
import pandas as pd
import numpy as np

from src.features.string_features import (
    jaccard_similarity,
    token_overlap_ratio,
    levenshtein_similarity,
    jaro_winkler_similarity
)


def compute_name_features_dict(s1_rec: Dict[str, Any], cand_rec: Dict[str, Any]) -> Dict[str, Any]:
    """Computes all name-based pairwise features for a single candidate pair."""
    name1_norm = str(s1_rec.get('business_name_norm', '') or '').strip()
    name2_norm = str(cand_rec.get('business_name_norm', '') or '').strip()

    name1_nosuff = str(s1_rec.get('business_name_no_suffix', '') or '').strip()
    name2_nosuff = str(cand_rec.get('business_name_no_suffix', '') or '').strip()

    s1_suff = str(s1_rec.get('business_name_suffix', '') or '').strip()
    cand_suff = str(cand_rec.get('business_name_suffix', '') or '').strip()

    # 1. Exact matches
    name_exact_norm = 1 if (name1_norm and name2_norm and name1_norm == name2_norm) else 0
    name_exact_no_suffix = 1 if (name1_nosuff and name2_nosuff and name1_nosuff == name2_nosuff) else 0

    # 2. Token sets
    tok1_str = str(s1_rec.get('business_name_significant_tokens', '') or '')
    tok2_str = str(cand_rec.get('business_name_significant_tokens', '') or '')

    tok1_set = set(tok1_str.split()) if tok1_str else set()
    tok2_set = set(tok2_str.split()) if tok2_str else set()

    name_token_jaccard = jaccard_similarity(tok1_set, tok2_set)
    name_token_overlap = token_overlap_ratio(tok1_set, tok2_set)

    # 3. Trigram sets
    tri1_str = str(s1_rec.get('business_name_char_trigrams', '') or '')
    tri2_str = str(cand_rec.get('business_name_char_trigrams', '') or '')

    tri1_set = set(tri1_str.split()) if tri1_str else set()
    tri2_set = set(tri2_str.split()) if tri2_str else set()

    name_char_trigram_jaccard = jaccard_similarity(tri1_set, tri2_set)

    # 4. Levenshtein and Jaro-Winkler
    name_levenshtein = levenshtein_similarity(name1_norm, name2_norm)
    name_jaro_winkler = jaro_winkler_similarity(name1_norm, name2_norm)

    # 5. Length features
    len1 = len(name1_norm)
    len2 = len(name2_norm)
    name_length_diff = abs(len1 - len2)
    name_length_ratio = (min(len1, len2) / max(len1, len2)) if max(len1, len2) > 0 else 1.0

    # 6. Suffix features
    s1_has_suffix = 1 if s1_suff else 0
    candidate_has_suffix = 1 if cand_suff else 0

    if s1_suff and cand_suff:
        suffix_exact_match = 1 if s1_suff == cand_suff else 0
    else:
        suffix_exact_match = -1

    # 7. Missingness
    s1_name_missing = 1 if not name1_norm else 0
    candidate_name_missing = 1 if not name2_norm else 0

    # 8. Token counts
    cnt1 = len(tok1_set)
    cnt2 = len(tok2_set)
    name_token_count_difference = abs(cnt1 - cnt2)
    name_token_count_ratio = (min(cnt1, cnt2) / max(cnt1, cnt2)) if max(cnt1, cnt2) > 0 else 1.0

    return {
        'name_exact_norm': name_exact_norm,
        'name_exact_no_suffix': name_exact_no_suffix,
        'name_token_jaccard': round(name_token_jaccard, 6),
        'name_token_overlap': round(name_token_overlap, 6),
        'name_char_trigram_jaccard': round(name_char_trigram_jaccard, 6),
        'name_levenshtein': round(name_levenshtein, 6),
        'name_jaro_winkler': round(name_jaro_winkler, 6),
        'name_len_s1': len1,
        'name_len_candidate': len2,
        'name_length_diff': name_length_diff,
        'name_length_ratio': round(name_length_ratio, 6),
        'suffix_exact_match': suffix_exact_match,
        's1_has_suffix': s1_has_suffix,
        'candidate_has_suffix': candidate_has_suffix,
        's1_name_missing': s1_name_missing,
        'candidate_name_missing': candidate_name_missing,
        'name_token_count_difference': name_token_count_difference,
        'name_token_count_ratio': round(name_token_count_ratio, 6)
    }
