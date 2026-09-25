"""
Unit tests for normalization and preprocessing module (src/preprocessing/normalization.py).
"""

import sys
import os
import unittest
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import (
    normalize_unicode_and_casing,
    normalize_business_name_str,
    normalize_business_address_str,
    generate_char_trigrams,
    process_dataframe_normalization
)


class TestNormalizationModule(unittest.TestCase):

    def test_unicode_and_casing(self):
        text = "Société GÉnÉrale & Cie"
        normalized = normalize_unicode_and_casing(text)
        self.assertEqual(normalized, "societe generale & cie")

    def test_name_normalization_suffixes(self):
        # Test 1: Inc
        norm, no_suf, suf = normalize_business_name_str("GOOGLE INC.")
        self.assertEqual(norm, "google inc")
        self.assertEqual(no_suf, "google")
        self.assertEqual(suf, "inc")

        # Test 2: French SARL
        norm, no_suf, suf = normalize_business_name_str("Boulangerie Pierre SARL")
        self.assertEqual(norm, "boulangerie pierre sarl")
        self.assertEqual(no_suf, "boulangerie pierre")
        self.assertEqual(suf, "sarl")

        # Test 3: Ampersand normalization
        norm, no_suf, suf = normalize_business_name_str("Johnson & Johnson Co.")
        self.assertEqual(norm, "johnson and johnson co")
        self.assertEqual(no_suf, "johnson and johnson")
        self.assertEqual(suf, "co")

    def test_address_normalization(self):
        # Test US Address with Zip, Street, Suite abbreviations
        raw_addr = "123 Main St., Ste 400, New York, NY 10001"
        norm, pin, st_num, city_loc = normalize_business_address_str(raw_addr)
        
        self.assertIn("123 main street suite 400 new york ny 10001", norm)
        self.assertEqual(pin, "10001")
        self.assertEqual(st_num, "123")

        # Test Indian Address with PIN and landmark
        raw_addr2 = "Plot 45, MG Road, Near Station, Bangalore 560001"
        norm2, pin2, st_num2, city_loc2 = normalize_business_address_str(raw_addr2)
        
        self.assertIn("plot 45 mg road near station bangalore 560001", norm2)
        self.assertEqual(pin2, "560001")
        self.assertEqual(st_num2, "45")

    def test_char_trigrams(self):
        trigrams = generate_char_trigrams("abcde")
        self.assertEqual(trigrams, ["abc", "bcd", "cde"])

        short_trigrams = generate_char_trigrams("ab")
        self.assertEqual(short_trigrams, ["ab"])

    def test_dataframe_processing(self):
        df = pd.DataFrame([
            {
                'entity_id': 'S1-100',
                'business_name': 'Acme Corp.',
                'business_address': '100 Broadway St, NY 10005',
                'country': 'US'
            }
        ])

        processed = process_dataframe_normalization(df)

        # Check preservation of raw fields
        self.assertEqual(processed['entity_id'].iloc[0], 'S1-100')
        self.assertEqual(processed['business_name'].iloc[0], 'Acme Corp.')
        self.assertEqual(processed['business_address'].iloc[0], '100 Broadway St, NY 10005')
        self.assertEqual(processed['country'].iloc[0], 'US')

        # Check new normalized fields
        self.assertEqual(processed['business_name_norm'].iloc[0], 'acme corp')
        self.assertEqual(processed['business_name_no_suffix'].iloc[0], 'acme')
        self.assertEqual(processed['business_name_suffix'].iloc[0], 'corp')
        self.assertEqual(processed['pin_code'].iloc[0], '10005')
        self.assertTrue(processed['has_address'].iloc[0])
        self.assertTrue(processed['has_pin'].iloc[0])


if __name__ == '__main__':
    unittest.main()
