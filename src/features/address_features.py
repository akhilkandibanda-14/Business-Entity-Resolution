"""
Address and location pairwise feature extraction module for Phase 3.
"""

from typing import Dict, Any
import pandas as pd
import numpy as np

from src.features.string_features import (
    jaccard_similarity,
    token_overlap_ratio,
    levenshtein_similarity
)


def compute_address_features_dict(s1_rec: Dict[str, Any], cand_rec: Dict[str, Any]) -> Dict[str, Any]:
    """Computes all address-based pairwise features for a single candidate pair."""
    addr1_norm = str(s1_rec.get('business_address_norm', '') or '').strip()
    addr2_norm = str(cand_rec.get('business_address_norm', '') or '').strip()

    c1 = str(s1_rec.get('country', '') or '').strip().upper()
    c2 = str(cand_rec.get('country', '') or '').strip().upper()

    pin1 = str(s1_rec.get('pin_code', '') or '').strip()
    pin2 = str(cand_rec.get('pin_code', '') or '').strip()

    st1 = str(s1_rec.get('street_number', '') or '').strip()
    st2 = str(cand_rec.get('street_number', '') or '').strip()

    city1_str = str(s1_rec.get('city_locality_tokens', '') or '').strip()
    city2_str = str(cand_rec.get('city_locality_tokens', '') or '').strip()

    # 1. Exact address
    address_exact_norm = 1 if (addr1_norm and addr2_norm and addr1_norm == addr2_norm) else 0

    # 2. Token sets
    tok1_str = str(s1_rec.get('address_tokens', '') or '')
    tok2_str = str(cand_rec.get('address_tokens', '') or '')

    tok1_set = set(tok1_str.split()) if tok1_str else set()
    tok2_set = set(tok2_str.split()) if tok2_str else set()

    address_token_jaccard = jaccard_similarity(tok1_set, tok2_set)
    address_token_overlap = token_overlap_ratio(tok1_set, tok2_set)

    # 3. Trigram sets
    tri1_str = str(s1_rec.get('address_char_trigrams', '') or '')
    tri2_str = str(cand_rec.get('address_char_trigrams', '') or '')

    tri1_set = set(tri1_str.split()) if tri1_str else set()
    tri2_set = set(tri2_str.split()) if tri2_str else set()

    address_char_trigram_jaccard = jaccard_similarity(tri1_set, tri2_set)

    # 4. Levenshtein
    address_levenshtein = levenshtein_similarity(addr1_norm, addr2_norm)

    # 5. Street number match
    if st1 and st2:
        street_number_match = 1 if st1 == st2 else 0
    else:
        street_number_match = -1

    # 6. PIN match
    if pin1 and pin2:
        pin_match = 1 if pin1 == pin2 else 0
    else:
        pin_match = -1

    # 7. Country match
    if c1 and c2:
        country_exact_match = 1 if c1 == c2 else 0
    else:
        country_exact_match = 0

    # 8. City / Locality features
    city1_set = set(city1_str.split()) if city1_str else set()
    city2_set = set(city2_str.split()) if city2_str else set()

    city_locality_jaccard = jaccard_similarity(city1_set, city2_set)
    city_locality_overlap = token_overlap_ratio(city1_set, city2_set)
    city_locality_exact = 1 if (city1_str and city2_str and city1_str == city2_str) else 0

    # 9. Presence & Missingness
    s1_has_address = 1 if addr1_norm else 0
    candidate_has_address = 1 if addr2_norm else 0

    s1_address_missing = 1 if not addr1_norm else 0
    candidate_address_missing = 1 if not addr2_norm else 0

    s1_pin_missing = 1 if not pin1 else 0
    candidate_pin_missing = 1 if not pin2 else 0

    s1_city_missing = 1 if not city1_str else 0
    candidate_city_missing = 1 if not city2_str else 0

    # 10. Length and count features
    len1 = len(addr1_norm)
    len2 = len(addr2_norm)
    address_length_difference = abs(len1 - len2)
    address_length_ratio = (min(len1, len2) / max(len1, len2)) if max(len1, len2) > 0 else 1.0

    cnt1 = len(tok1_set)
    cnt2 = len(tok2_set)
    address_token_count_difference = abs(cnt1 - cnt2)
    address_token_count_ratio = (min(cnt1, cnt2) / max(cnt1, cnt2)) if max(cnt1, cnt2) > 0 else 1.0

    return {
        'address_exact_norm': address_exact_norm,
        'address_token_jaccard': round(address_token_jaccard, 6),
        'address_token_overlap': round(address_token_overlap, 6),
        'address_char_trigram_jaccard': round(address_char_trigram_jaccard, 6),
        'address_levenshtein': round(address_levenshtein, 6),
        'street_number_match': street_number_match,
        'pin_match': pin_match,
        'country_exact_match': country_exact_match,
        'city_locality_jaccard': round(city_locality_jaccard, 6),
        'city_locality_overlap': round(city_locality_overlap, 6),
        'city_locality_exact': city_locality_exact,
        's1_has_address': s1_has_address,
        'candidate_has_address': candidate_has_address,
        's1_address_missing': s1_address_missing,
        'candidate_address_missing': candidate_address_missing,
        's1_pin_missing': s1_pin_missing,
        'candidate_pin_missing': candidate_pin_missing,
        's1_city_missing': s1_city_missing,
        'candidate_city_missing': candidate_city_missing,
        'address_length_difference': address_length_difference,
        'address_length_ratio': round(address_length_ratio, 6),
        'address_token_count_difference': address_token_count_difference,
        'address_token_count_ratio': round(address_token_count_ratio, 6)
    }
