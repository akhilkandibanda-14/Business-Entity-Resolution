"""
Low-level string and token set similarity utilities for Phase 3 Feature Engineering.
Calculates Jaccard, Token Overlap, Levenshtein, and Jaro-Winkler similarities safely.
"""

from typing import Set, Union, List, Optional
import numpy as np


def jaccard_similarity(s1_tokens: Set[str], s2_tokens: Set[str]) -> float:
    """Calculates Jaccard similarity between two token/trigram sets."""
    if not s1_tokens or not s2_tokens:
        return 0.0
    intersection = len(s1_tokens & s2_tokens)
    union = len(s1_tokens | s2_tokens)
    return float(intersection / union) if union > 0 else 0.0


def token_overlap_ratio(s1_tokens: Set[str], s2_tokens: Set[str]) -> float:
    """Calculates token overlap ratio: intersection / min(len1, len2)."""
    if not s1_tokens or not s2_tokens:
        return 0.0
    intersection = len(s1_tokens & s2_tokens)
    min_len = min(len(s1_tokens), len(s2_tokens))
    return float(intersection / min_len) if min_len > 0 else 0.0


def levenshtein_distance(str1: str, str2: str) -> int:
    """Computes exact edit distance between two strings using DP."""
    if str1 == str2:
        return 0
    len1, len2 = len(str1), len(str2)
    if len1 == 0:
        return len2
    if len2 == 0:
        return len1

    # Single-row DP space optimization
    current_row = list(range(len2 + 1))
    for i in range(1, len1 + 1):
        previous_row = current_row
        current_row = [i] + [0] * len2
        ch1 = str1[i - 1]
        for j in range(1, len2 + 1):
            add = previous_row[j] + 1
            delete = current_row[j - 1] + 1
            change = previous_row[j - 1] + (0 if ch1 == str2[j - 1] else 1)
            current_row[j] = min(add, delete, change)

    return current_row[len2]


def levenshtein_similarity(str1: str, str2: str) -> float:
    """Computes normalized edit similarity: 1.0 - distance / max(len1, len2)."""
    if not str1 and not str2:
        return 1.0
    if not str1 or not str2:
        return 0.0
    if str1 == str2:
        return 1.0
    max_len = max(len(str1), len(str2))
    dist = levenshtein_distance(str1, str2)
    return float(1.0 - (dist / max_len)) if max_len > 0 else 0.0


def jaro_winkler_similarity(s1: str, s2: str, p: float = 0.1) -> float:
    """
    Computes Jaro-Winkler string similarity.
    Range: [0.0, 1.0]
    """
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0

    len1, len2 = len(s1), len(s2)
    max_dist = max(len1, len2) // 2 - 1
    if max_dist < 0:
        max_dist = 0

    s1_matches = [False] * len1
    s2_matches = [False] * len2

    matches = 0
    for i in range(len1):
        start = max(0, i - max_dist)
        end = min(i + max_dist + 1, len2)
        for j in range(start, end):
            if not s2_matches[j] and s1[i] == s2[j]:
                s1_matches[i] = True
                s2_matches[j] = True
                matches += 1
                break

    if matches == 0:
        return 0.0

    t = 0
    k = 0
    for i in range(len1):
        if s1_matches[i]:
            while not s2_matches[k]:
                k += 1
            if s1[i] != s2[k]:
                t += 1
            k += 1
    t /= 2.0

    jaro = (matches / len1 + matches / len2 + (matches - t) / matches) / 3.0

    # Prefix scale
    prefix = 0
    for i in range(min(len1, len2, 4)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break

    return float(jaro + prefix * p * (1.0 - jaro))
