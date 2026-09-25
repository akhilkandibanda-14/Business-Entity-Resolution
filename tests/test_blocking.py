"""
Unit test suite for Phase 2 blocking and candidate generation module.
"""

import unittest
import pandas as pd
import numpy as np

from src.blocking.country_blocker import CountryBlocker, analyze_country_agreement
from src.blocking.exact_name_blocker import ExactNameBlocker, SortedTokenBlocker
from src.blocking.token_blocker import TokenBlocker
from src.blocking.pin_blocker import PinBlocker
from src.blocking.char_ngram_blocker import CharNgramBlocker
from src.blocking.candidate_generator import CandidateGenerator
from src.blocking.blocking_metrics import calculate_blocking_metrics


class TestBlockingPipeline(unittest.TestCase):

    def setUp(self):
        # Sample test dataframes
        self.df_s1 = pd.DataFrame([
            {
                'entity_id': 'S1-1',
                'country': 'US',
                'business_name': 'Acme Corp',
                'business_name_norm': 'acme corp',
                'business_name_no_suffix': 'acme',
                'business_name_suffix': 'corp',
                'business_name_tokens': 'acme corp',
                'business_name_significant_tokens': 'acme',
                'business_name_char_trigrams': 'acm cme cor orp',
                'pin_code': '90210'
            },
            {
                'entity_id': 'S1-2',
                'country': 'India',
                'business_name': 'Global Tech Pvt Ltd',
                'business_name_norm': 'global tech pvt ltd',
                'business_name_no_suffix': 'global tech',
                'business_name_suffix': 'pvt ltd',
                'business_name_tokens': 'global tech pvt ltd',
                'business_name_significant_tokens': 'global tech',
                'business_name_char_trigrams': 'glo lob oba bal tec ech',
                'pin_code': '500081'
            },
            {
                'entity_id': 'S1-3',
                'country': 'US',
                'business_name': 'Empty Match Co',
                'business_name_norm': 'empty match co',
                'business_name_no_suffix': 'empty match',
                'business_name_suffix': 'co',
                'business_name_tokens': 'empty match co',
                'business_name_significant_tokens': 'empty match',
                'business_name_char_trigrams': 'emp mpt pty mat atc tch',
                'pin_code': ''
            }
        ])

        self.df_s2 = pd.DataFrame([
            {
                'entity_id': 'S2-101',
                'country': 'US',
                'business_name': 'ACME Inc',
                'business_name_norm': 'acme inc',
                'business_name_no_suffix': 'acme',
                'business_name_suffix': 'inc',
                'business_name_tokens': 'acme inc',
                'business_name_significant_tokens': 'acme',
                'business_name_char_trigrams': 'acm cme inc ncp',
                'pin_code': '90210'
            },
            {
                'entity_id': 'S2-102',
                'country': 'India',
                'business_name': 'Tech Global Private Limited',
                'business_name_norm': 'tech global private limited',
                'business_name_no_suffix': 'tech global',
                'business_name_suffix': 'private limited',
                'business_name_tokens': 'tech global private limited',
                'business_name_significant_tokens': 'tech global',
                'business_name_char_trigrams': 'tec ech glo lob oba bal',
                'pin_code': '500081'
            }
        ])

        self.df_s3 = pd.DataFrame([
            {
                'entity_id': 'S3-201',
                'country': 'US',
                'business_name': 'Acme Corp',
                'business_name_norm': 'acme corp',
                'business_name_no_suffix': 'acme',
                'business_name_suffix': 'corp',
                'business_name_tokens': 'acme corp',
                'business_name_significant_tokens': 'acme',
                'business_name_char_trigrams': 'acm cme cor orp',
                'pin_code': '90210'
            }
        ])

    def test_1_exact_normalized_name_match(self):
        blocker = ExactNameBlocker(use_country_partition=True)
        cand = blocker.generate_candidates(self.df_s1, self.df_s2, self.df_s3)
        
        # S1-1 exact norm matches S3-201
        norm_matches = cand[cand['blocker'] == 'exact_name_norm']
        self.assertIn(('S1-1', 'S3-201'), zip(norm_matches['source1_entity_id'], norm_matches['candidate_entity_id']))
        
        # S1-1 no_suffix "acme" matches S2-101 and S3-201
        nosuff_matches = cand[cand['blocker'] == 'exact_name_no_suffix']
        self.assertIn(('S1-1', 'S2-101'), zip(nosuff_matches['source1_entity_id'], nosuff_matches['candidate_entity_id']))

    def test_2_sorted_token_match(self):
        blocker = SortedTokenBlocker(use_country_partition=True)
        cand = blocker.generate_candidates(self.df_s1, self.df_s2, self.df_s3)
        
        # S1-2 "global tech" matches S2-102 "tech global"
        self.assertTrue(len(cand) > 0)
        self.assertIn(('S1-2', 'S2-102'), zip(cand['source1_entity_id'], cand['candidate_entity_id']))

    def test_3_token_inverted_index(self):
        blocker = TokenBlocker(max_df_ratio=1.0, max_df_absolute=100, use_country_partition=True)
        cand, stats = blocker.generate_candidates(self.df_s1, self.df_s2, self.df_s3)
        
        self.assertIn('vocab_size', stats)
        self.assertIn(('S1-1', 'S2-101'), zip(cand['source1_entity_id'], cand['candidate_entity_id']))

    def test_4_pin_match(self):
        blocker = PinBlocker(use_country_partition=True)
        cand = blocker.generate_candidates(self.df_s1, self.df_s2, self.df_s3)
        
        # S1-1 PIN 90210 matches S2-101 and S3-201
        self.assertIn(('S1-1', 'S2-101'), zip(cand['source1_entity_id'], cand['candidate_entity_id']))

    def test_5_char_ngram_match(self):
        blocker = CharNgramBlocker(min_shared_trigrams=2, max_df_ratio=1.0, max_df_absolute=100, use_country_partition=True)
        cand, stats = blocker.generate_candidates(self.df_s1, self.df_s2, self.df_s3)
        
        self.assertIn('vocab_size', stats)
        self.assertTrue(len(cand) > 0)

    def test_6_union_and_deduplication(self):
        generator = CandidateGenerator(use_country_partition=True)
        cand, stats = generator.generate_all_candidates(self.df_s1, self.df_s2, self.df_s3)
        
        # Check no duplicate (s1_id, candidate_id) rows exist
        pairs = list(zip(cand['source1_entity_id'], cand['candidate_entity_id']))
        self.assertEqual(len(pairs), len(set(pairs)))

    def test_7_blocker_provenance(self):
        generator = CandidateGenerator(use_country_partition=True)
        cand, _ = generator.generate_all_candidates(self.df_s1, self.df_s2, self.df_s3)
        
        # S1-1 -> S3-201 matches on exact_name_norm, exact_name_no_suffix, pin, etc.
        s1_s3_match = cand[(cand['source1_entity_id'] == 'S1-1') & (cand['candidate_entity_id'] == 'S3-201')]
        self.assertEqual(len(s1_s3_match), 1)
        prov = s1_s3_match.iloc[0]['blockers']
        self.assertIn('|', prov) # Multiple blockers concatenated

    def test_8_missing_values(self):
        # Pass DataFrames with empty strings / NaNs
        df_empty_s1 = self.df_s1.copy()
        df_empty_s1['business_name_norm'] = ""
        df_empty_s1['pin_code'] = np.nan
        
        blocker = ExactNameBlocker(use_country_partition=True)
        cand = blocker.generate_candidates(df_empty_s1, self.df_s2, self.df_s3)
        # Should not crash and handle gracefully
        self.assertIsInstance(cand, pd.DataFrame)

    def test_9_different_countries(self):
        # S1-1 is US, S2-102 is India -> should NOT match under country partition
        generator = CandidateGenerator(use_country_partition=True)
        cand, _ = generator.generate_all_candidates(self.df_s1, self.df_s2, self.df_s3)
        
        cross_country = cand[(cand['source1_entity_id'] == 'S1-1') & (cand['candidate_entity_id'] == 'S2-102')]
        self.assertEqual(len(cross_country), 0)

    def test_10_multiple_matches_for_one_s1(self):
        generator = CandidateGenerator(use_country_partition=True)
        cand, _ = generator.generate_all_candidates(self.df_s1, self.df_s2, self.df_s3)
        
        s1_cand_count = len(cand[cand['source1_entity_id'] == 'S1-1'])
        self.assertGreaterEqual(s1_cand_count, 2) # Matches S2-101 and S3-201

    def test_11_zero_match_s1(self):
        generator = CandidateGenerator(use_country_partition=True)
        cand, _ = generator.generate_all_candidates(self.df_s1, self.df_s2, self.df_s3)
        
        s1_3_cand = cand[cand['source1_entity_id'] == 'S1-3']
        self.assertEqual(len(s1_3_cand), 0)

    def test_12_no_cartesian_product(self):
        # Verify candidate count is far smaller than total Cartesian product size
        generator = CandidateGenerator(use_country_partition=True)
        cand, _ = generator.generate_all_candidates(self.df_s1, self.df_s2, self.df_s3)
        
        cartesian_size = len(self.df_s1) * (len(self.df_s2) + len(self.df_s3))
        self.assertLess(len(cand), cartesian_size)

    def test_13_metrics_calculation(self):
        generator = CandidateGenerator(use_country_partition=True)
        cand, _ = generator.generate_all_candidates(self.df_s1, self.df_s2, self.df_s3)
        
        gt_df = pd.DataFrame([
            {'source1_entity_id': 'S1-1', 'matched_entity_ids': 'S2-101,S3-201'},
            {'source1_entity_id': 'S1-2', 'matched_entity_ids': 'S2-102'},
            {'source1_entity_id': 'S1-3', 'matched_entity_ids': ''}
        ])
        
        metrics = calculate_blocking_metrics(
            candidate_df=cand,
            gt_df=gt_df,
            df_s1=self.df_s1,
            df_s2=self.df_s2,
            df_s3=self.df_s3,
            use_country_partition=True
        )
        
        self.assertIn('blocking_recall_pct', metrics)
        self.assertIn('complete_entity_recall_pct', metrics)
        self.assertEqual(metrics['blocking_recall_pct'], 100.0)


if __name__ == '__main__':
    unittest.main()
