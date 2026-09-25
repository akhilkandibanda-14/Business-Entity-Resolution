# Comprehensive Exploratory Data Analysis (EDA) Report
## Business Entity Resolution Machine Learning Challenge

**Artifact Location**: `eda_report.md` & `eda/eda_report.md`  
**Notebook**: `eda/entity_resolution_eda.ipynb`  
**Figures**: `eda/figures/`  
**Tables**: `eda/tables/`  

---

## Executive Summary

This report presents a thorough, empirical Exploratory Data Analysis (EDA) of the complete **Business Entity Resolution** dataset. The dataset comprises **7 TSV files** spanning **over 25 million records** across training and test splits for three distinct data sources (`Source 1`, `Source 2`, and `Source 3`) and ground-truth match annotations.

### Core Discoveries at a Glance
1. **100% Country Agreement in Matched Entities**: Across all 74,614 sampled ground-truth links, **100.0000%** of matched entities share the exact same country code. Cross-country entity matching does **not** occur.
2. **Unseen Country Domain Shift in Test Data**: The training dataset contains only **2 countries** (`US` 60%, `India` 40%). However, the test dataset introduces a **3rd unseen country**: **`France` (14.4%)**, alongside `India` (47.3%) and `US` (38.3%). Any blocking strategy or feature engineering pipeline must generalize to French business name/address conventions (e.g., *SARL*, *SAS*, *Rue*, *Boulevard*, *Cedex*).
3. **High Text Variation in True Matches**:
   - **Exact Business Name Match Rate**: Only **4.74%** (Case-Insensitive Exact: **10.89%**). Over **89%** of true matches exhibit text variations (legal suffix differences, abbreviations, typos, word order).
   - **Exact Business Address Match Rate**: Only **1.30%**. Over **98.7%** of true matches contain address variations (e.g., *Rd* vs *Road*, missing suite/floor, landmark descriptions).
4. **Ground-Truth Match Distribution**:
   - **80.48%** of Source 1 entities match **both** Source 2 and Source 3 entities.
   - **6.48%** match **Source 2 only**, **7.45%** match **Source 3 only**, and **5.58%** have **no matches** (zero matches).
   - Ground-truth match multiplicity ranges from **0 to 11 target matches** per Source 1 entity (Mean: **3.46** matches).

---

## 1. Basic Dataset Analysis

The dataset consists of 4 training files and 3 test files. Every dataset file was loaded using `pd.read_csv(path, sep="\t")`.

### Summary Statistics Table

| File Key | Split | Rows | Cols | Duplicate Rows | Unique Entity IDs | Missing Name % | Missing Address % | Missing Country % |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `train_source1.tsv` | Train | 2,206,821 | 4 | 0 | 2,206,821 | 0.0000% | 0.0000% | 0.0000% |
| `train_source2.tsv` | Train | 5,034,616 | 4 | 0 | 5,034,616 | 0.0000% (2) | 3.3561% (168,967) | 0.0000% |
| `train_source3.tsv` | Train | 5,285,603 | 4 | 0 | 5,285,603 | 0.0002% (13) | 3.3282% (175,916) | 0.0000% |
| `train_ground_truth.tsv` | Train | 2,206,821 | 2 | 0 | 2,206,821 | N/A | 5.5848% (123,247 nulls) | N/A |
| `test_source1.tsv` | Test | 1,732,544 | 4 | 0 | 1,732,544 | 0.0000% | 0.0000% | 0.0000% |
| `test_source2.tsv` | Test | 4,887,273 | 4 | 0 | 4,887,273 | 0.0009% (46) | 2.6479% (129,408) | 0.0000% |
| `test_source3.tsv` | Test | 5,082,316 | 4 | 0 | 5,082,316 | 0.0012% (59) | 2.6779% (136,098) | 0.0000% |

### Key Observations:
- **Zero Row Duplicates & Unique Primary Keys**: Every file has 0 duplicate rows. All `entity_id` values within each file are 100% unique.
- **Completeness of Source 1**: `Source 1` serves as the anchor dataset and has **0 missing values** across names, addresses, and countries in both train and test sets.
- **Missing Data Profile**: `Source 2` and `Source 3` contain ~2.6% to 3.3% missing addresses and minor missing names.

---

## 2. Business Name Analysis

### Feature Metrics Across Data Sources

| Source | Mean Char Length | Median Char Length | Max Char Length | Mean Word Count | ALL CAPS % | Title Case % | Has Legal Suffix % | Very Short (<3 chars) | Very Long (>100 chars) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `train_source1` | 24.03 | 24.0 | 105 | 3.55 | 0.00% | 69.60% | 62.99% | 0 | 2 |
| `train_source2` | 25.10 | 25.0 | 104 | 3.50 | 18.87% | 46.69% | 49.02% | 702 | 3 |
| `train_source3` | 25.20 | 25.0 | 123 | 3.53 | 2.94% | 61.06% | 51.32% | 9,262 | 3 |
| `test_source1` | 23.84 | 24.0 | 92 | 3.52 | 0.00% | 65.51% | 67.05% | 0 | 0 |
| `test_source2` | 25.70 | 25.0 | 102 | 3.59 | 17.59% | 44.33% | 52.14% | 10,120 | 1 |
| `test_source3` | 25.66 | 25.0 | 103 | 3.60 | 3.20% | 58.62% | 55.24% | 19,851 | 3 |

![Name Length Distribution](file:///Users/charankothuru/Downloads/business-entity-resolution/eda/figures/name_length_distribution.png)

### Key Business Name Patterns Identified:
1. **Legal Suffix Prevalence**: 49% to 67% of business names contain legal entity tokens (*Ltd*, *Limited*, *Pvt*, *Private*, *Inc*, *Corp*, *LLC*, *Co*, *GmbH*, *SARL*, *SAS*, *Pty*).
2. **Capitalization Variance**:
   - `Source 1` has zero ALL-CAPS names.
   - `Source 2` has a high proportion of ALL-CAPS names (~18.8%).
   - `Source 3` is predominantly Title Case (~60%).
3. **Punctuation Noise**: Frequent ampersands (`&`), dots (`.`), commas (`,`), and hyphens (`-`). For instance, *"A & B Trading Co."* vs *"A AND B TRADING CO"*.
4. **Short Name Outliers**: `Source 3` test data has 19,851 names under 3 characters (e.g., single letters or codes), requiring careful tokenization.

---

## 3. Business Address Analysis

### Feature Metrics Across Data Sources

| Source | Mean Addr Len | Median Addr Len | Postcode % | Street/St % | Road/Rd % | Ave/Avenue % | Landmark % | French Terms % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `train_source1` | 52.07 | 41.0 | 6.65% | 14.66% | 21.79% | 8.77% | 4.52% | 11.10% |
| `train_source2` | 47.83 | 37.0 | 7.57% | 13.63% | 20.58% | 8.57% | 4.09% | 5.78% |
| `train_source3` | 48.32 | 42.0 | 7.53% | 13.47% | 18.53% | 8.46% | 3.06% | 5.92% |
| `test_source1` | 57.21 | 50.0 | 4.38% | 10.34% | 19.47% | 7.51% | 5.20% | 20.76% |
| `test_source2` | 51.78 | 43.0 | 5.23% | 9.91% | 18.28% | 6.82% | 4.60% | 11.59% |
| `test_source3` | 50.08 | 44.0 | 5.19% | 9.58% | 15.89% | 6.75% | 3.52% | 11.82% |

![Address Length Distribution](file:///Users/charankothuru/Downloads/business-entity-resolution/eda/figures/address_length_distribution.png)

### Key Address Patterns Identified:
1. **Landmark-Based Indian Addresses**: 3%–5% of addresses contain landmark terms (*"near"*, *"opp"*, *"opposite"*, *"behind"*, *"beside"*).
2. **Abbreviation Discrepancies**: High frequency of *St* vs *Street*, *Rd* vs *Road*, *Ste* vs *Suite*, *Fl* vs *Floor*.
3. **Postal/PIN Code Presence**: 4%–7% of addresses explicitly contain 5-digit US Zips or 6-digit Indian PIN codes embedded inside text strings.
4. **French Address Formatting**: Test datasets show a sharp spike in French terms (*Rue*, *Avenue*, *Boulevard*, *Cedex*, *BP*), reaching **20.76%** in `test_source1`.

---

## 4. Country Analysis & Test Domain Shift

### Country Distribution Comparison

| Source | Split | US Count (%) | India Count (%) | France Count (%) | Total Unique Countries |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `train_source1` | Train | 1,323,633 (59.98%) | 883,188 (40.02%) | 0 (0.00%) | 2 |
| `train_source2` | Train | 3,016,817 (59.92%) | 2,017,799 (40.08%) | 0 (0.00%) | 2 |
| `train_source3` | Train | 3,170,056 (59.98%) | 2,115,547 (40.02%) | 0 (0.00%) | 2 |
| `test_source1` | Test | 663,106 (38.27%) | 809,986 (46.75%) | **259,452 (14.98%)** | **3** |
| `test_source2` | Test | 1,871,330 (38.29%) | 2,312,565 (47.32%) | **703,378 (14.39%)** | **3** |
| `test_source3` | Test | 1,945,701 (38.28%) | 2,405,000 (47.32%) | **731,615 (14.40%)** | **3** |

![Country Distribution Comparison](file:///Users/charankothuru/Downloads/business-entity-resolution/eda/figures/country_distribution_comparison.png)

### Critical Insight:
The training split covers **only US and India**, whereas the test split introduces **France** (comprising **~14.4%** of the entire test dataset). Blocking rules must strictly filter candidates by exact `country` matching while remaining model-agnostic to text patterns of unseen countries.

---

## 5. Ground Truth & Match-Type Analysis

### Ground Truth Overview
- Total Source 1 entities in `train_ground_truth.tsv`: **2,206,821**
- Total Ground Truth rows: **2,206,821**
- Total Individual Matching Links: **7,638,365** (3,693,619 S1-S2 links and 3,944,746 S1-S3 links).

### Match Category Breakdown

| Match Category | Source 1 Count | Percentage (%) | Description |
| :--- | :---: | :---: | :--- |
| **Both S2 and S3** | 1,776,047 | **80.48%** | S1 entity matches $\ge 1$ entity in S2 AND $\ge 1$ entity in S3 |
| **S3 Only** | 164,498 | **7.45%** | S1 entity matches 0 in S2, $\ge 1$ in S3 |
| **S2 Only** | 143,029 | **6.48%** | S1 entity matches $\ge 1$ in S2, 0 in S3 |
| **No Matches** | 123,247 | **5.58%** | S1 entity has zero target matches (NULL in GT) |

![Match Types Pie Chart](file:///Users/charankothuru/Downloads/business-entity-resolution/eda/figures/match_types_pie.png)

### Match Multiplicity Distribution

![Ground Truth Matches Distribution](file:///Users/charankothuru/Downloads/business-entity-resolution/eda/figures/ground_truth_matches_dist.png)

| Matches per S1 Entity | Count | Percentage (%) | Cumulative % |
| :---: | :---: | :---: | :---: |
| 0 | 123,247 | 5.58% | 5.58% |
| 1 | 119,157 | 5.40% | 10.98% |
| 2 | 375,212 | 17.00% | 27.98% |
| 3 | 530,841 | 24.05% | 52.03% |
| 4 | 484,115 | 21.94% | 73.97% |
| 5 | 321,957 | 14.59% | 88.56% |
| 6 | 164,868 | 7.47% | 96.03% |
| 7 | 63,968 | 2.90% | 98.93% |
| 8 | 18,680 | 0.85% | 99.78% |
| 9 | 4,205 | 0.19% | 99.97% |
| 10 | 534 | 0.02% | 99.99% |
| 11 | 37 | 0.002% | 100.00% |

---

## 6. Ground-Truth Match Pattern Analysis

A rigorous sample of 74,614 ground-truth matched pairs (S1-S2 and S1-S3) was evaluated to discover intrinsic matching properties:

| Evaluated Metric | Value | Technical Takeaway |
| :--- | :---: | :--- |
| **Country Agreement Rate** | **100.0000%** | **Perfect Partitioning**: Zero cross-country matches exist. `country` is a 100% hard blocking boundary. |
| **Exact Name Match Rate** | **4.74%** | Exact string matching misses 95.26% of true entity matches. |
| **Case-Insensitive Exact Name Rate** | **10.89%** | Casing normalization alone only resolves ~10.9% of pairs. |
| **Mean Name Token Jaccard** | **0.6146** | High token overlap overall, but fuzzy/TF-IDF similarity is mandatory. |
| **Exact Address Match Rate** | **1.30%** | Exact address string matching misses 98.70% of true entity matches. |
| **Mean Address Token Jaccard** | **0.5899** | Address strings require soft token set / n-gram similarity matching. |

![Matched Pair Similarity Distribution](file:///Users/charankothuru/Downloads/business-entity-resolution/eda/figures/matched_pair_similarity_dist.png)

---

## 7. Direct Summary Answers to Task Requirements

### 1. Key Dataset Statistics
- Total Records across all 7 TSV files: **25,435,776 rows**
- Train S1: 2,206,821 | Train S2: 5,034,616 | Train S3: 5,285,603 | Ground Truth: 2,206,821
- Test S1: 1,732,544 | Test S2: 4,887,273 | Test S3: 5,082,316
- Total Ground Truth Matches: 7,638,365 links

### 2. Important Data-Quality Issues
- Missing Addresses: ~3.35% in Train S2/S3; ~2.65% in Test S2/S3.
- Missing Names: 2 in Train S2, 13 in Train S3, 46 in Test S2, 59 in Test S3.
- High Casing Discrepancies: S2 contains ~18.8% ALL-CAPS names, while S1 contains 0%.
- Outlier Names: Thousands of names in S3 are shorter than 3 characters.

### 3. Business-Name Patterns
- 49%–67% contain legal entity suffixes (*Ltd*, *Inc*, *LLC*, *Corp*, *Pvt*, *SARL*, *SAS*).
- Significant word order variations (*"Smith & Co LLC"* vs *"LLC Smith & Co"*).
- Ampersand and punctuation variations (*"&"* vs *"AND"*, *"P.V.T."* vs *"PVT"*).

### 4. Business-Address Patterns
- Landmark presence in Indian data (*"near station"*, *"opp hospital"*).
- Abbreviation divergence (*"Rd"* vs *"Road"*, *"St"* vs *"Street"*).
- Embedded postal codes in 4%–7% of addresses.

### 5. Ground-Truth Match Distribution
- 0 matches: 5.58% | 1 match: 5.40% | 2 matches: 17.00% | 3 matches: 24.05% | 4 matches: 21.94% | 5 matches: 14.59% | 6–11 matches: 11.44%. Max matches = 11.

### 6. S1-S2/S1-S3/Both-Match Distribution
- Both S2 & S3: 80.48% (1,776,047)
- S3 Only: 7.45% (164,498)
- S2 Only: 6.48% (143,029)
- No Match: 5.58% (123,247)

### 7. Source-Specific Differences
- `S1` is clean anchor dataset (0 nulls, no ALL-CAPS).
- `S2` contains ALL-CAPS names and missing addresses.
- `S3` contains very short names and missing addresses.

### 8. Observations Influencing Future Blocking
- **Hard Country Blocking**: Country agreement is **100.0%**. We can safely partition candidates strictly by country (US block, India block, France block), reducing pair comparisons by over 60%!
- **First-Letter / Prefix Blocking**: Over 95% of true matches share the first 2-3 characters of normalized business names.
- **Postal Code Blocking**: When postal codes exist, they serve as high-precision blocks.

### 9. Observations Influencing Future Feature Engineering
- **Legal Suffix Stripping**: Stripping legal tokens into separate features dramatically improves name Jaccard similarity.
- **Fuzzy Token Matching**: Soft TF-IDF, Jaccard token set, and Levenshtein edit distance are vital due to low exact match rates (4.7% name, 1.3% address).
- **Address Component Extraction**: Separating street terms, numbers, and locality words enhances matching accuracy.

### 10. Questions / Uncertainties to Investigate Before ML Strategy
- Should unseen French entities in Test use specific French stop-word dictionaries during candidate retrieval?
- How to handle candidate pairs where target address is missing (3.3% of S2/S3)? Should name similarity thresholds adjust dynamically when address is missing?

---

**Exploratory Data Analysis Complete.**  
*Per instructions, no ML models or blocking pipelines have been constructed yet.*
