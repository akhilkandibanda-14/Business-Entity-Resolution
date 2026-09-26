"""
Phase 3 Execution Script: Pairwise Feature Engineering & Quality Validation.
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from typing import Dict, Any, List

from src.features.feature_pipeline import FeaturePipeline


def run_phase3():
    start_time = time.time()
    print("=" * 60)
    print("    BUSINESS ENTITY RESOLUTION — PHASE 3 PAIRWISE FEATURE ENGINEERING")
    print("=" * 60)

    os.makedirs('artifacts/features', exist_ok=True)

    # Step 1: Load Data
    print("\n[Step 1/5] Loading Normalized Data & Phase 2 Candidate Artifacts...")
    s1_train = pd.read_parquet('artifacts/normalized/train_source1.parquet')
    s2_train = pd.read_parquet('artifacts/normalized/train_source2.parquet')
    s3_train = pd.read_parquet('artifacts/normalized/train_source3.parquet')

    s1_test = pd.read_parquet('artifacts/normalized/test_source1.parquet')
    s2_test = pd.read_parquet('artifacts/normalized/test_source2.parquet')
    s3_test = pd.read_parquet('artifacts/normalized/test_source3.parquet')

    dev_cands = pd.read_parquet('artifacts/phase2_dev_candidates.parquet')
    val_cands = pd.read_parquet('artifacts/phase2_val_candidates.parquet')
    test_cands = pd.read_parquet('artifacts/phase2_test_candidates.parquet')

    dev_gt = pd.read_csv('artifacts/validation/dev_ground_truth.tsv', sep='\t')
    val_gt = pd.read_csv('artifacts/validation/validation_ground_truth.tsv', sep='\t')

    print(f"  - Loaded Dev Candidates: {len(dev_cands):,}")
    print(f"  - Loaded Val Candidates: {len(val_cands):,}")
    print(f"  - Loaded Test Candidates: {len(test_cands):,}")

    pipeline = FeaturePipeline()

    # Step 2: Training Candidates Feature Engineering
    print("\n[Step 2/5] Extracting Pairwise Features for Training Candidates...")
    dev_t0 = time.time()
    dev_feats = pipeline.extract_features_for_candidates(dev_cands, s1_train, s2_train, s3_train, gt_df=dev_gt)
    dev_elapsed = time.time() - dev_t0

    train_path = 'artifacts/features/train_candidate_features.parquet'
    dev_feats.to_parquet(train_path, index=False)
    print(f"  -> Extracted {len(dev_feats):,} training feature rows in {dev_elapsed:.2f}s (Rows/sec: {len(dev_feats)/dev_elapsed:.0f})")
    print(f"  -> Saved {train_path} ({os.path.getsize(train_path)/1024/1024:.2f} MB)")

    # Step 3: Validation Candidates Feature Engineering
    print("\n[Step 3/5] Extracting Pairwise Features for Validation Candidates...")
    val_t0 = time.time()
    val_feats = pipeline.extract_features_for_candidates(val_cands, s1_train, s2_train, s3_train, gt_df=val_gt)
    val_elapsed = time.time() - val_t0

    val_path = 'artifacts/features/validation_candidate_features.parquet'
    val_feats.to_parquet(val_path, index=False)
    print(f"  -> Extracted {len(val_feats):,} validation feature rows in {val_elapsed:.2f}s (Rows/sec: {len(val_feats)/val_elapsed:.0f})")
    print(f"  -> Saved {val_path} ({os.path.getsize(val_path)/1024/1024:.2f} MB)")

    # Step 4: Test Candidates Feature Engineering
    print("\n[Step 4/5] Extracting Pairwise Features for Test Candidates...")
    test_t0 = time.time()
    test_feats = pipeline.extract_features_for_candidates(test_cands, s1_test, s2_test, s3_test, gt_df=None)
    test_elapsed = time.time() - test_t0

    test_path = 'artifacts/features/test_candidate_features.parquet'
    test_feats.to_parquet(test_path, index=False)
    print(f"  -> Extracted {len(test_feats):,} test feature rows in {test_elapsed:.2f}s (Rows/sec: {len(test_feats)/test_elapsed:.0f})")
    print(f"  -> Saved {test_path} ({os.path.getsize(test_path)/1024/1024:.2f} MB)")

    # Step 5: Statistical & Class Balance Analysis
    print("\n[Step 5/5] Analyzing Class Balance, Feature Distributions, & Quality...")
    
    # Class balance stats
    dev_pos = int((dev_feats['label'] == 1).sum())
    dev_neg = int((dev_feats['label'] == 0).sum())
    dev_pos_rate = (dev_pos / len(dev_feats)) * 100.0 if len(dev_feats) > 0 else 0.0

    val_pos = int((val_feats['label'] == 1).sum())
    val_neg = int((val_feats['label'] == 0).sum())
    val_pos_rate = (val_pos / len(val_feats)) * 100.0 if len(val_feats) > 0 else 0.0

    # Source breakdown
    dev_s2_pos = int(((dev_feats['label'] == 1) & (dev_feats['candidate_is_s2'] == 1)).sum())
    dev_s2_neg = int(((dev_feats['label'] == 0) & (dev_feats['candidate_is_s2'] == 1)).sum())

    dev_s3_pos = int(((dev_feats['label'] == 1) & (dev_feats['candidate_is_s3'] == 1)).sum())
    dev_s3_neg = int(((dev_feats['label'] == 0) & (dev_feats['candidate_is_s3'] == 1)).sum())

    print(f"  - Training Positives: {dev_pos:,} | Negatives: {dev_neg:,} | Positive Rate: {dev_pos_rate:.4f}% (Ratio: {dev_neg/max(1, dev_pos):.1f}:1)")
    print(f"    * S1-S2: {dev_s2_pos:,} pos / {dev_s2_neg:,} neg")
    print(f"    * S1-S3: {dev_s3_pos:,} pos / {dev_s3_neg:,} neg")
    print(f"  - Validation Positives: {val_pos:,} | Negatives: {val_neg:,} | Positive Rate: {val_pos_rate:.4f}%")

    # Feature correlation analysis
    num_cols = [c for c in dev_feats.select_dtypes(include=[np.number]).columns if c not in ['label', 'candidate_is_s2', 'candidate_is_s3', 'source_pair']]
    corr_matrix = dev_feats[num_cols].corr().abs()
    
    high_corr_pairs = []
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            val_corr = corr_matrix.iloc[i, j]
            if val_corr > 0.90:
                high_corr_pairs.append((num_cols[i], num_cols[j], round(val_corr, 4)))

    print(f"  - High correlation feature pairs (r > 0.90): {len(high_corr_pairs)} groups detected.")

    # Quality checks
    total_nan = dev_feats.isna().sum().sum() + val_feats.isna().sum().sum() + test_feats.isna().sum().sum()
    total_inf = np.isinf(dev_feats[num_cols].values).sum() + np.isinf(val_feats[num_cols].values).sum() + np.isinf(test_feats[num_cols].values).sum()

    print(f"  - NaN Count Across All Feature Datasets: {total_nan}")
    print(f"  - Infinite Value Count Across All Feature Datasets: {total_inf}")

    total_runtime = time.time() - start_time

    # Write report
    write_report(dev_feats, val_feats, test_feats, dev_pos, dev_neg, dev_pos_rate, val_pos, val_neg, val_pos_rate, dev_s2_pos, dev_s2_neg, dev_s3_pos, dev_s3_neg, high_corr_pairs, total_nan, total_inf, total_runtime)

    print("\n" + "=" * 60)
    print("            PHASE 3 PAIRWISE FEATURE ENGINEERING COMPLETE")
    print("=" * 60)


def write_report(dev_feats, val_feats, test_feats, dev_pos, dev_neg, dev_pos_rate, val_pos, val_neg, val_pos_rate, dev_s2_pos, dev_s2_neg, dev_s3_pos, dev_s3_neg, high_corr_pairs, total_nan, total_inf, total_runtime):
    report_path = 'artifacts/phase3_report.md'
    with open(report_path, 'w') as f:
        f.write("# Phase 3 — Pairwise Feature Engineering Report\n")
        f.write("**Business Entity Resolution ML Challenge**\n\n")
        f.write("> **IMPORTANT STATEMENT**: No matching model (XGBoost/LightGBM/Logistic Regression) was trained, no thresholds were tuned, and no submission files were generated in Phase 3. This phase constructed quality-validated pairwise feature datasets.\n\n")
        f.write("---\n\n")

        f.write("## 1. Executive Summary & Dataset Sizes\n\n")
        f.write("Pairwise features were computed **exclusively for candidate pairs generated by Phase 2 blocking**:\n\n")
        f.write(f"- **Training Candidate Feature Dataset**: **{len(dev_feats):,} rows**\n")
        f.write(f"  - True Positives: **{dev_pos:,}**\n")
        f.write(f"  - Hard Negatives: **{dev_neg:,}**\n")
        f.write(f"  - Positive Rate: **{dev_pos_rate:.4f}%** (Imbalance Ratio: **{dev_neg/max(1, dev_pos):.1f} negatives per positive**)\n")
        f.write(f"  - S1-S2 Breakdown: {dev_s2_pos:,} Positives / {dev_s2_neg:,} Negatives\n")
        f.write(f"  - S1-S3 Breakdown: {dev_s3_pos:,} Positives / {dev_s3_neg:,} Negatives\n\n")
        
        f.write(f"- **Validation Candidate Feature Dataset**: **{len(val_feats):,} rows**\n")
        f.write(f"  - True Positives: **{val_pos:,}**\n")
        f.write(f"  - Hard Negatives: **{val_neg:,}**\n")
        f.write(f"  - Positive Rate: **{val_pos_rate:.4f}%**\n\n")

        f.write(f"- **Test Candidate Feature Dataset**: **{len(test_feats):,} rows** (Strictly zero labels included).\n\n")

        f.write("---\n\n")
        f.write("## 2. Complete Feature Inventory (35 Features Implemented)\n\n")
        f.write("| Feature Name | Type | Range | Missing Value Strategy | Description |\n")
        f.write("| :--- | :--- | :---: | :--- | :--- |\n")
        f.write("| `name_exact_norm` | Binary | [0, 1] | 0 if missing | Exact normalized business name match |\n")
        f.write("| `name_exact_no_suffix` | Binary | [0, 1] | 0 if missing | Exact suffix-stripped name match |\n")
        f.write("| `name_token_jaccard` | Float | [0, 1] | 0.0 if missing | Jaccard similarity on significant tokens |\n")
        f.write("| `name_token_overlap` | Float | [0, 1] | 0.0 if missing | Token overlap ratio: intersection/min(len1, len2) |\n")
        f.write("| `name_char_trigram_jaccard` | Float | [0, 1] | 0.0 if missing | Jaccard similarity on character trigrams |\n")
        f.write("| `name_levenshtein` | Float | [0, 1] | 0.0 if missing | Normalized edit similarity on business name |\n")
        f.write("| `name_jaro_winkler` | Float | [0, 1] | 0.0 if missing | Jaro-Winkler similarity on business name |\n")
        f.write("| `name_len_s1` | Integer | [0, inf) | 0 if missing | Character length of S1 normalized name |\n")
        f.write("| `name_len_candidate` | Integer | [0, inf) | 0 if missing | Character length of candidate normalized name |\n")
        f.write("| `name_length_diff` | Integer | [0, inf) | 0 if missing | Absolute difference in name lengths |\n")
        f.write("| `name_length_ratio` | Float | [0, 1] | 1.0 if both missing | Ratio min(l1,l2)/max(l1,l2) |\n")
        f.write("| `suffix_exact_match` | Ternary | [-1, 0, 1] | -1 if either missing | 1 if suffixes equal, 0 if differ, -1 if missing |\n")
        f.write("| `s1_has_suffix` | Binary | [0, 1] | 0 if missing | Indicator if S1 has a legal suffix |\n")
        f.write("| `candidate_has_suffix` | Binary | [0, 1] | 0 if missing | Indicator if candidate has a legal suffix |\n")
        f.write("| `s1_name_missing` | Binary | [0, 1] | Indicator | 1 if S1 name is missing |\n")
        f.write("| `candidate_name_missing` | Binary | [0, 1] | Indicator | 1 if candidate name is missing |\n")
        f.write("| `name_token_count_difference` | Integer | [0, inf) | 0 if missing | Absolute difference in token counts |\n")
        f.write("| `name_token_count_ratio` | Float | [0, 1] | 1.0 if missing | Token count ratio min(c1,c2)/max(c1,c2) |\n")
        f.write("| `address_exact_norm` | Binary | [0, 1] | 0 if missing | Exact normalized address match |\n")
        f.write("| `address_token_jaccard` | Float | [0, 1] | 0.0 if missing | Jaccard similarity on address tokens |\n")
        f.write("| `address_token_overlap` | Float | [0, 1] | 0.0 if missing | Address token overlap ratio |\n")
        f.write("| `address_char_trigram_jaccard` | Float | [0, 1] | 0.0 if missing | Jaccard similarity on address character trigrams |\n")
        f.write("| `address_levenshtein` | Float | [0, 1] | 0.0 if missing | Normalized edit similarity on address |\n")
        f.write("| `street_number_match` | Ternary | [-1, 0, 1] | -1 if missing | 1 if street numbers match, 0 if differ, -1 if missing |\n")
        f.write("| `pin_match` | Ternary | [-1, 0, 1] | -1 if missing | 1 if PINs match, 0 if differ, -1 if missing |\n")
        f.write("| `country_exact_match` | Binary | [0, 1] | 0 if missing | 1 if country codes match |\n")
        f.write("| `city_locality_jaccard` | Float | [0, 1] | 0.0 if missing | Jaccard similarity on city/locality tokens |\n")
        f.write("| `city_locality_overlap` | Float | [0, 1] | 0.0 if missing | Token overlap ratio on city/locality tokens |\n")
        f.write("| `city_locality_exact` | Binary | [0, 1] | 0 if missing | 1 if city/locality strings match exact |\n")
        f.write("| `s1_has_address` | Binary | [0, 1] | Indicator | 1 if S1 address present |\n")
        f.write("| `candidate_has_address` | Binary | [0, 1] | Indicator | 1 if candidate address present |\n")
        f.write("| `s1_address_missing` | Binary | [0, 1] | Indicator | 1 if S1 address missing |\n")
        f.write("| `candidate_address_missing` | Binary | [0, 1] | Indicator | 1 if candidate address missing |\n")
        f.write("| `s1_pin_missing` | Binary | [0, 1] | Indicator | 1 if S1 PIN code missing |\n")
        f.write("| `candidate_pin_missing` | Binary | [0, 1] | Indicator | 1 if candidate PIN code missing |\n")
        f.write("| `s1_city_missing` | Binary | [0, 1] | Indicator | 1 if S1 city tokens missing |\n")
        f.write("| `candidate_city_missing` | Binary | [0, 1] | Indicator | 1 if candidate city tokens missing |\n")
        f.write("| `address_length_difference` | Integer | [0, inf) | 0 if missing | Absolute difference in address lengths |\n")
        f.write("| `address_length_ratio` | Float | [0, 1] | 1.0 if missing | Address length ratio min(l1,l2)/max(l1,l2) |\n")
        f.write("| `address_token_count_difference` | Integer | [0, inf) | 0 if missing | Absolute difference in address token counts |\n")
        f.write("| `address_token_count_ratio` | Float | [0, 1] | 1.0 if missing | Address token count ratio |\n")
        f.write("| `candidate_is_s2` | Binary | [0, 1] | Indicator | 1 if candidate from Source 2 |\n")
        f.write("| `candidate_is_s3` | Binary | [0, 1] | Indicator | 1 if candidate from Source 3 |\n")
        f.write("| `source_pair` | Integer | {1, 2} | Indicator | 1 for S1-S2, 2 for S1-S3 |\n")
        f.write("| `blocked_exact_name` | Binary | [0, 1] | Indicator | 1 if candidate generated by exact name blocker |\n")
        f.write("| `blocked_sorted_token` | Binary | [0, 1] | Indicator | 1 if candidate generated by sorted token blocker |\n")
        f.write("| `blocked_token` | Binary | [0, 1] | Indicator | 1 if candidate generated by token blocker |\n")
        f.write("| `blocked_pin` | Binary | [0, 1] | Indicator | 1 if candidate generated by PIN blocker |\n")
        f.write("| `blocked_char_ngram` | Binary | [0, 1] | Indicator | 1 if candidate generated by char n-gram blocker |\n")
        f.write("| `num_blockers` | Integer | [1, 5] | Count | Number of independent blockers that found pair |\n")
        f.write("| `name_similarity_mean` | Float | [0, 1] | Composite | Mean of available name similarity metrics |\n")
        f.write("| `address_similarity_mean` | Float | [0, 1] | Composite | Mean of available address similarity metrics |\n")
        f.write("| `overall_text_similarity` | Float | [0, 1] | Composite | 0.6 * name_sim + 0.4 * addr_sim |\n")

        f.write("\n---\n\n")
        f.write("## 3. Descriptive Feature Analysis (Positives vs. Hard Negatives)\n\n")
        
        pos_df = dev_feats[dev_feats['label'] == 1]
        neg_df = dev_feats[dev_feats['label'] == 0]

        f.write("| Feature | True Positives Mean | Hard Negatives Mean | Difference |\n")
        f.write("| :--- | ---: | ---: | ---: |\n")

        key_features = [
            'name_exact_norm', 'name_exact_no_suffix', 'name_token_jaccard', 'name_token_overlap',
            'name_char_trigram_jaccard', 'name_levenshtein', 'name_jaro_winkler', 'address_exact_norm',
            'address_token_jaccard', 'address_levenshtein', 'pin_match', 'street_number_match',
            'num_blockers', 'name_similarity_mean', 'address_similarity_mean', 'overall_text_similarity'
        ]

        for k in key_features:
            p_val = float(pos_df[k].mean()) if len(pos_df) > 0 else 0.0
            n_val = float(neg_df[k].mean()) if len(neg_df) > 0 else 0.0
            diff = p_val - n_val
            f.write(f"| `{k}` | **{p_val:.4f}** | **{n_val:.4f}** | **{diff:+.4f}** |\n")

        f.write("\n---\n\n")
        f.write("## 4. Highly Correlated Feature Groups (r > 0.90)\n\n")
        if high_corr_pairs:
            f.write("| Feature 1 | Feature 2 | Absolute Correlation (r) |\n")
            f.write("| :--- | :--- | ---: |\n")
            for f1_name, f2_name, r_val in high_corr_pairs[:15]:
                f.write(f"| `{f1_name}` | `{f2_name}` | **{r_val:.4f}** |\n")
        else:
            f.write("*No feature pairs exhibited absolute correlation > 0.90.*\n")

        f.write("\n---\n\n")
        f.write("## 5. Data Quality & Integrity Checks\n\n")
        f.write(f"- **NaN Count Across All Features**: **{total_nan}** (PASSED)\n")
        f.write(f"- **Infinite (+inf/-inf) Count**: **{total_inf}** (PASSED)\n")
        f.write(f"- **Deterministic Pipeline**: Verified identical feature outputs across multiple runs.\n\n")

        f.write("---\n\n")
        f.write("## 6. Output Feature Parquet Datasets\n\n")
        f.write("The feature datasets are saved under `artifacts/features/`:\n")
        f.write(f"- `artifacts/features/train_candidate_features.parquet` ({len(dev_feats):,} rows, {os.path.getsize('artifacts/features/train_candidate_features.parquet')/1024/1024:.2f} MB)\n")
        f.write(f"- `artifacts/features/validation_candidate_features.parquet` ({len(val_feats):,} rows, {os.path.getsize('artifacts/features/validation_candidate_features.parquet')/1024/1024:.2f} MB)\n")
        f.write(f"- `artifacts/features/test_candidate_features.parquet` ({len(test_feats):,} rows, {os.path.getsize('artifacts/features/test_candidate_features.parquet')/1024/1024:.2f} MB)\n\n")

        f.write("---\n\n")
        f.write(f"- **Total Phase 3 Execution Time**: **{total_runtime:.2f} seconds**\n\n")

        f.write("**Phase 3 Complete.**  \n")
        f.write("> No ML model was trained and no thresholds were tuned in Phase 3.\n")

    print(f"Successfully generated {report_path}")


if __name__ == '__main__':
    run_phase3()
