"""
Feature engineering package for Business Entity Resolution.
"""

from src.features.string_features import (
    jaccard_similarity,
    token_overlap_ratio,
    levenshtein_similarity,
    jaro_winkler_similarity
)
from src.features.name_features import compute_name_features_dict
from src.features.address_features import compute_address_features_dict
from src.features.pair_features import compute_pair_features_dict
from src.features.feature_pipeline import FeaturePipeline

__all__ = [
    'jaccard_similarity',
    'token_overlap_ratio',
    'levenshtein_similarity',
    'jaro_winkler_similarity',
    'compute_name_features_dict',
    'compute_address_features_dict',
    'compute_pair_features_dict',
    'FeaturePipeline'
]
