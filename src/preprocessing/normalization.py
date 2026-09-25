"""
Vectorized Normalization and Preprocessing Module for Business Entity Resolution.
Provides reusable, highly optimized pandas functions for business name and address
normalization, tokenization, character trigram generation, and structural field extraction.
"""

import re
import unicodedata
import pandas as pd
import numpy as np
from typing import List, Set, Dict, Tuple, Any, Optional

# Compiled Regular Expressions for Legal Suffixes & Address Pattern Extraction
LEGAL_SUFFIXES_ORDERED = [
    'incorporated', 'corporation', 'private', 'limited', 'company',
    'gmbh', 'sarl', 'eurl', 'corp', 'ltd', 'pvt', 'llc', 'llp',
    'pty', 'inc', 'sas', 'plc', 'spa', 'bv', 'nv', 'co', 'sa'
]

# Capturing group pattern for pandas str.extract
SUFFIX_EXTRACT_PATTERN = re.compile(r'(\b(?:' + '|'.join(LEGAL_SUFFIXES_ORDERED) + r')\b)', re.IGNORECASE)
SUFFIX_REPLACE_PATTERN = re.compile(r'\b(?:' + '|'.join(LEGAL_SUFFIXES_ORDERED) + r')\b', re.IGNORECASE)

# Address Abbreviation Patterns
ADDRESS_ABBREVIATIONS = [
    (re.compile(r'\brd\b', re.IGNORECASE), 'road'),
    (re.compile(r'\bst\b', re.IGNORECASE), 'street'),
    (re.compile(r'\bave\b', re.IGNORECASE), 'avenue'),
    (re.compile(r'\bste\b', re.IGNORECASE), 'suite'),
    (re.compile(r'\bfl\b', re.IGNORECASE), 'floor'),
    (re.compile(r'\bapt\b', re.IGNORECASE), 'apartment'),
    (re.compile(r'\bbldg\b', re.IGNORECASE), 'building'),
    (re.compile(r'\bhwy\b', re.IGNORECASE), 'highway'),
    (re.compile(r'\bblvd\b', re.IGNORECASE), 'boulevard'),
    (re.compile(r'\bp\.?o\.?\s*box\b', re.IGNORECASE), 'po box')
]

PIN_REGEX_COMPILED = re.compile(r'(\b\d{5,6}\b)')
STREET_NUM_COMPILED = re.compile(r'(^\d+[a-z]?\b|\b\d{1,5}\b)')

SIGNIFICANT_STOPWORDS = {
    'and', 'the', 'of', 'for', 'in', 'on', 'at', 'to', 'a', 'an', 'is',
    'co', 'corp', 'inc', 'ltd', 'llc', 'pvt', 'limited', 'company', 'corporation',
    'gmbh', 'sarl', 'sas', 'sa', 'bv', 'nv', 'pty', 'plc', 'private'
}


def normalize_unicode_and_casing(text: str) -> str:
    """Lowercase text and normalize Unicode characters to ASCII (NFKD)."""
    if not isinstance(text, str) or not text:
        return ""
    nfkd_form = unicodedata.normalize('NFKD', text)
    ascii_text = nfkd_form.encode('ASCII', 'ignore').decode('utf-8')
    return ascii_text.lower().strip()


def generate_char_trigrams(text: str) -> List[str]:
    """Generate character trigrams list from text."""
    if not isinstance(text, str) or not text:
        return []
    if len(text) < 3:
        return [text]
    return [text[i:i+3] for i in range(len(text) - 2)]


def generate_char_trigrams_str(text: str) -> str:
    """Generate space-joined character trigrams string from text."""
    if not isinstance(text, str) or not text:
        return ""
    if len(text) < 3:
        return text
    return " ".join([text[i:i+3] for i in range(len(text) - 2)])


def normalize_business_name_str(raw_name: str) -> Tuple[str, str, str]:
    """Single-string helper for name normalization."""
    if not isinstance(raw_name, str) or not raw_name.strip():
        return "", "", ""
    name = normalize_unicode_and_casing(raw_name)
    name = name.replace('&', ' and ')
    name = re.sub(r'[^\w\s\-]', ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    
    if not name:
        return "", "", ""
        
    match = SUFFIX_REPLACE_PATTERN.search(name)
    suffix = match.group(0).lower() if match else ""
    no_suffix = SUFFIX_REPLACE_PATTERN.sub('', name).strip() if suffix else name
    no_suffix = re.sub(r'\s+', ' ', no_suffix).strip()
    return name, no_suffix if no_suffix else name, suffix


def normalize_business_address_str(raw_address: str) -> Tuple[str, str, str, str]:
    """Single-string helper for address normalization."""
    if not isinstance(raw_address, str) or not raw_address.strip():
        return "", "", "", ""
    addr = normalize_unicode_and_casing(raw_address)
    
    pin_m = re.search(r'\b\d{5,6}\b', addr)
    pin_code = pin_m.group(0) if pin_m else ""
    
    st_m = re.search(r'^\d+[a-z]?\b|\b\d{1,5}\b', addr)
    street_num = st_m.group(0) if st_m else ""
    
    for p_reg, repl in ADDRESS_ABBREVIATIONS:
        addr = p_reg.sub(repl, addr)
        
    addr_clean = re.sub(r'[^\w\s\-]', ' ', addr)
    addr_clean = re.sub(r'\s+', ' ', addr_clean).strip()
    
    toks = addr_clean.split()
    city_loc = " ".join(toks[-3:]) if len(toks) >= 3 else (" ".join(toks[-2:]) if len(toks) >= 2 else addr_clean)
    return addr_clean, pin_code, street_num, city_loc


def process_dataframe_normalization(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies high-speed vectorized preprocessing and normalization pipeline to a DataFrame.
    Preserves all original columns without modification.
    """
    processed = df.copy()
    
    # -------------------------------------------------------------
    # 1. Business Name Normalization
    # -------------------------------------------------------------
    raw_names = processed['business_name'].fillna("").astype(str)
    
    # Lowercase & ampersands
    norm_names = raw_names.str.lower().str.replace('&', ' and ', regex=False)
    # Remove non-alphanumeric/non-hyphen punctuation
    norm_names = norm_names.str.replace(r'[^\w\s\-]', ' ', regex=True).str.replace(r'\s+', ' ', regex=True).str.strip()
    
    processed['business_name_norm'] = norm_names
    
    # Extract suffix using precompiled regex
    extracted_suffix = norm_names.str.extract(SUFFIX_EXTRACT_PATTERN, expand=False).fillna("").str.lower()
    processed['business_name_suffix'] = extracted_suffix
    
    # Remove suffix using precompiled regex
    no_suffix = norm_names.str.replace(SUFFIX_REPLACE_PATTERN, '', regex=True).str.replace(r'\s+', ' ', regex=True).str.strip()
    processed['business_name_no_suffix'] = np.where(no_suffix == "", norm_names, no_suffix)
    
    # Tokens & Character Features
    name_tokens_list = norm_names.str.split()
    no_suf_tokens = processed['business_name_no_suffix'].str.split()
    
    processed['business_name_tokens'] = norm_names
    processed['business_name_significant_tokens'] = no_suf_tokens.apply(
        lambda toks: " ".join([w for w in toks if w not in SIGNIFICANT_STOPWORDS]) if isinstance(toks, list) else ""
    )
    processed['business_name_char_trigrams'] = norm_names.apply(generate_char_trigrams_str)
    
    processed['name_char_length'] = norm_names.str.len()
    processed['name_token_count'] = name_tokens_list.apply(lambda toks: len(toks) if isinstance(toks, list) else 0)
    processed['name_has_digits'] = norm_names.str.contains(r'\d', regex=True)
    
    # -------------------------------------------------------------
    # 2. Business Address Normalization
    # -------------------------------------------------------------
    raw_addrs = processed['business_address'].fillna("").astype(str)
    has_addr = (raw_addrs != "") & (raw_addrs.str.strip() != "")
    processed['has_address'] = has_addr
    
    addr_lower = raw_addrs.str.lower()
    
    # PIN Code using precompiled regex
    pins = addr_lower.str.extract(PIN_REGEX_COMPILED, expand=False).fillna("")
    processed['pin_code'] = pins
    processed['has_pin'] = pins != ""
    
    # Street Number using precompiled regex
    street_nums = addr_lower.str.extract(STREET_NUM_COMPILED, expand=False).fillna("")
    processed['street_number'] = street_nums
    
    # Expand abbreviations
    addr_exp = addr_lower
    for p_reg, repl in ADDRESS_ABBREVIATIONS:
        addr_exp = addr_exp.str.replace(p_reg, repl, regex=True)
        
    addr_norm = addr_exp.str.replace(r'[^\w\s\-]', ' ', regex=True).str.replace(r'\s+', ' ', regex=True).str.strip()
    processed['business_address_norm'] = addr_norm
    
    addr_tokens_list = addr_norm.str.split()
    processed['address_tokens'] = addr_norm
    processed['address_char_length'] = addr_norm.str.len()
    processed['address_token_count'] = addr_tokens_list.apply(lambda toks: len(toks) if isinstance(toks, list) else 0)
    processed['address_char_trigrams'] = addr_norm.apply(generate_char_trigrams_str)
    
    # Inferred City/Locality Tokens (trailing 2-3 tokens stored as string)
    processed['city_locality_tokens'] = addr_tokens_list.apply(
        lambda toks: " ".join(toks[-3:]) if isinstance(toks, list) and len(toks) >= 3 else (" ".join(toks[-2:]) if isinstance(toks, list) and len(toks) >= 2 else (" ".join(toks) if isinstance(toks, list) else ""))
    )
    
    return processed
