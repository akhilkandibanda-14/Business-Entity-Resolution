"""
Unit test suite for Phase 3 feature engineering module.
"""

import unittest
import pandas as pd
import numpy as np

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


class TestFeatureEngineering(unittest.TestCase):

    def setUp(self):
        self.s1_rec = {
            'entity_id': 'S1-1',
            'country': 'US',
            'business_name': 'Acme Corp',
            'business_name_norm': 'acme corp',
            'business_name_no_suffix': 'acme',
            'business_name_suffix': 'corp',
            'business_name_tokens': 'acme corp',
            'business_name_significant_tokens': 'acme',
            'business_name_char_trigrams': 'acm cme cor orp',
            'business_address': '123 Main St',
            'business_address_norm': '123 main street',
            'address_tokens': '123 main street',
            'address_char_trigrams': '123 mai ain str tre eet',
            'pin_code': '90210',
            'street_number': '123',
            'city_locality_tokens': 'tucson az'
        }

        self.cand_rec_matching = {
            'entity_id': 'S2-101',
            'country': 'US',
            'business_name': 'Acme Inc',
            'business_name_norm': 'acme inc',
            'business_name_no_suffix': 'acme',
            'business_name_suffix': 'inc',
            'business_name_tokens': 'acme inc',
            'business_name_significant_tokens': 'acme',
            'business_name_char_trigrams': 'acm cme inc',
            'business_address': '123 Main St',
            'business_address_norm': '123 main street',
            'address_tokens': '123 main street',
            'address_char_trigrams': '123 mai ain str tre eet',
            'pin_code': '90210',
            'street_number': '123',
            'city_locality_tokens': 'tucson az'
        }

        self.cand_rec_differing = {
            'entity_id': 'S3-202',
            'country': 'US',
            'business_name': 'Zebra Robotics',
            'business_name_norm': 'zebra robotics',
            'business_name_no_suffix': 'zebra robotics',
            'business_name_suffix': '',
            'business_name_tokens': 'zebra robotics',
            'business_name_significant_tokens': 'zebra robotics',
            'business_name_char_trigrams': 'zeb ebr bra rob obt',
            'business_address': '456 Oak Rd',
            'business_address_norm': '456 oak road',
            'address_tokens': '456 oak road',
            'address_char_trigrams': '456 oak roa oad',
            'pin_code': '90211',
            'street_number': '456',
            'city_locality_tokens': 'phoenix az'
        }

    def test_1_exact_name_match(self):
        f1 = compute_name_features_dict(self.s1_rec, self.s1_rec)
        self.assertEqual(f1['name_exact_norm'], 1)
        
        f2 = compute_name_features_dict(self.s1_rec, self.cand_rec_matching)
        self.assertEqual(f2['name_exact_norm'], 0)

    def test_2_name_without_suffix_match(self):
        f = compute_name_features_dict(self.s1_rec, self.cand_rec_matching)
        self.assertEqual(f['name_exact_no_suffix'], 1)

    def test_3_token_jaccard(self):
        j = jaccard_similarity({'acme', 'corp'}, {'acme', 'inc'})
        self.assertAlmostEqual(j, 1/3, places=4)

    def test_4_token_overlap(self):
        ov = token_overlap_ratio({'acme'}, {'acme', 'robotics'})
        self.assertEqual(ov, 1.0)

    def test_5_character_trigram_similarity(self):
        f = compute_name_features_dict(self.s1_rec, self.cand_rec_matching)
        self.assertGreater(f['name_char_trigram_jaccard'], 0.0)

    def test_6_levenshtein_similarity(self):
        sim = levenshtein_similarity('acme corp', 'acme inc')
        self.assertGreater(sim, 0.5)

    def test_7_jaro_winkler_similarity(self):
        jw = jaro_winkler_similarity('acme corp', 'acme inc')
        self.assertGreater(jw, 0.8)

    def test_8_address_similarity(self):
        f = compute_address_features_dict(self.s1_rec, self.cand_rec_matching)
        self.assertEqual(f['address_exact_norm'], 1)
        self.assertEqual(f['address_token_jaccard'], 1.0)

    def test_9_pin_matching(self):
        f_match = compute_address_features_dict(self.s1_rec, self.cand_rec_matching)
        self.assertEqual(f_match['pin_match'], 1)

        f_diff = compute_address_features_dict(self.s1_rec, self.cand_rec_differing)
        self.assertEqual(f_diff['pin_match'], 0)

    def test_10_street_number_matching(self):
        f_match = compute_address_features_dict(self.s1_rec, self.cand_rec_matching)
        self.assertEqual(f_match['street_number_match'], 1)

        f_diff = compute_address_features_dict(self.s1_rec, self.cand_rec_differing)
        self.assertEqual(f_diff['street_number_match'], 0)

    def test_11_country_matching(self):
        f = compute_address_features_dict(self.s1_rec, self.cand_rec_matching)
        self.assertEqual(f['country_exact_match'], 1)

    def test_12_missing_values(self):
        empty_rec = {}
        f_name = compute_name_features_dict(empty_rec, empty_rec)
        self.assertEqual(f_name['s1_name_missing'], 1)
        self.assertEqual(f_name['candidate_name_missing'], 1)

        f_addr = compute_address_features_dict(empty_rec, empty_rec)
        self.assertEqual(f_addr['pin_match'], -1)
        self.assertEqual(f_addr['street_number_match'], -1)

    def test_13_suffix_handling(self):
        f_match = compute_name_features_dict(self.s1_rec, self.s1_rec)
        self.assertEqual(f_match['suffix_exact_match'], 1)

        f_diff = compute_name_features_dict(self.s1_rec, self.cand_rec_matching) # corp vs inc
        self.assertEqual(f_diff['suffix_exact_match'], 0)

    def test_14_length_ratios(self):
        f = compute_name_features_dict(self.s1_rec, self.cand_rec_matching)
        self.assertGreater(f['name_length_ratio'], 0.0)
        self.assertLessEqual(f['name_length_ratio'], 1.0)

    def test_15_blocker_provenance(self):
        name_f = compute_name_features_dict(self.s1_rec, self.cand_rec_matching)
        addr_f = compute_address_features_dict(self.s1_rec, self.cand_rec_matching)
        pair_f = compute_pair_features_dict('S2', 'exact_name_norm|pin', name_f, addr_f)
        
        self.assertEqual(pair_f['blocked_exact_name'], 1)
        self.assertEqual(pair_f['blocked_pin'], 1)
        self.assertEqual(pair_f['num_blockers'], 2)

    def test_16_label_generation_and_pipeline(self):
        df_s1 = pd.DataFrame([self.s1_rec])
        df_s2 = pd.DataFrame([self.cand_rec_matching])
        df_s3 = pd.DataFrame([self.cand_rec_differing])

        cand_df = pd.DataFrame([
            {'source1_entity_id': 'S1-1', 'candidate_entity_id': 'S2-101', 'candidate_source': 'S2', 'blockers': 'exact_name_norm'},
            {'source1_entity_id': 'S1-1', 'candidate_entity_id': 'S3-202', 'candidate_source': 'S3', 'blockers': 'token'}
        ])

        gt_df = pd.DataFrame([
            {'source1_entity_id': 'S1-1', 'matched_entity_ids': 'S2-101'}
        ])

        pipeline = FeaturePipeline()
        feats_df = pipeline.extract_features_for_candidates(cand_df, df_s1, df_s2, df_s3, gt_df=gt_df)

        self.assertEqual(len(feats_df), 2)
        self.assertIn('label', feats_df.columns)
        
        row_pos = feats_df[feats_df['candidate_entity_id'] == 'S2-101'].iloc[0]
        row_neg = feats_df[feats_df['candidate_entity_id'] == 'S3-202'].iloc[0]
        
        self.assertEqual(row_pos['label'], 1)
        self.assertEqual(row_neg['label'], 0)

    def test_17_no_nan_or_infinity(self):
        df_s1 = pd.DataFrame([self.s1_rec])
        df_s2 = pd.DataFrame([self.cand_rec_matching])
        df_s3 = pd.DataFrame([self.cand_rec_differing])

        cand_df = pd.DataFrame([
            {'source1_entity_id': 'S1-1', 'candidate_entity_id': 'S2-101', 'candidate_source': 'S2', 'blockers': 'pin'}
        ])

        pipeline = FeaturePipeline()
        feats_df = pipeline.extract_features_for_candidates(cand_df, df_s1, df_s2, df_s3)

        self.assertEqual(feats_df.isna().sum().sum(), 0)
        num_cols = feats_df.select_dtypes(include=[np.number]).columns
        self.assertEqual(np.isinf(feats_df[num_cols].values).sum(), 0)

    def test_18_duplicate_pair_handling(self):
        df_s1 = pd.DataFrame([self.s1_rec])
        df_s2 = pd.DataFrame([self.cand_rec_matching])
        df_s3 = pd.DataFrame([self.cand_rec_differing])

        cand_df = pd.DataFrame([
            {'source1_entity_id': 'S1-1', 'candidate_entity_id': 'S2-101', 'candidate_source': 'S2', 'blockers': 'pin'},
            {'source1_entity_id': 'S1-1', 'candidate_entity_id': 'S2-101', 'candidate_source': 'S2', 'blockers': 'exact_name_norm'}
        ])

        pipeline = FeaturePipeline()
        feats_df = pipeline.extract_features_for_candidates(cand_df.drop_duplicates(subset=['source1_entity_id', 'candidate_entity_id']), df_s1, df_s2, df_s3)
        self.assertEqual(len(feats_df), 1)


if __name__ == '__main__':
    unittest.main()
