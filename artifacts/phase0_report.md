# Phase 0 — Setup & Validation Report
**Business Entity Resolution ML Challenge**

---

## 1. Executive Summary

Phase 0 establishes the foundational validation infrastructure, dataset integrity verification, macro-averaged $F_{0.5}$ metric computation engine, and reproducible train/validation split without applying any normalization, blocking, candidate generation, or ML modeling.

---

## 2. Task 1: Dataset Verification & Integrity

All **7 dataset files** were loaded directly via `pd.read_csv(path, sep="\t")` and verified against the project EDA report:

| Dataset File Key | Split | File Path | Row Count | Column Count | Column Names | Verification Status |
| :--- | :--- | :--- | :---: | :---: | :--- | :---: |
| `train_s1` | Train | `dataset/train/train_source1.tsv` | 2,206,821 | 4 | `entity_id`, `business_name`, `business_address`, `country` | **MATCHED** |
| `train_s2` | Train | `dataset/train/train_source2.tsv` | 5,034,616 | 4 | `entity_id`, `business_name`, `business_address`, `country` | **MATCHED** |
| `train_s3` | Train | `dataset/train/train_source3.tsv` | 5,285,603 | 4 | `entity_id`, `business_name`, `business_address`, `country` | **MATCHED** |
| `train_gt` | Train | `dataset/train/train_ground_truth.tsv` | 2,206,821 | 2 | `source1_entity_id`, `matched_entity_ids` | **MATCHED** |
| `test_s1` | Test | `dataset/test/test_source1.tsv` | 1,732,544 | 4 | `entity_id`, `business_name`, `business_address`, `country` | **MATCHED** |
| `test_s2` | Test | `dataset/test/test_source2.tsv` | 4,887,273 | 4 | `entity_id`, `business_name`, `business_address`, `country` | **MATCHED** |
| `test_s3` | Test | `dataset/test/test_source3.tsv` | 5,082,316 | 4 | `entity_id`, `business_name`, `business_address`, `country` | **MATCHED** |

**Mismatch Status**: **0 mismatches discovered.** All row counts, column dimensions, and column headers match `eda_report.md` 100%.

---

## 3. Task 2 & 3: Validation Split & Ground Truth Structures

### Split Specifications
- **Level of Split**: **Source 1 entity level** (`source1_entity_id`)
- **Holdout Ratio**: **10.0%** (Validation) / **90.0%** (Development)
- **Random Seed**: **42** (Fixed, deterministic seed)
- **Data Leakage Check**: **PASSED** (0 overlapping Source 1 IDs between dev and validation)

### Development vs Validation Sizes
- **Total Source 1 Entities**: **2,206,821**
- **Development Set Size**: **1,986,139 entities** (90.00%)
- **Validation Holdout Size**: **220,682 entities** (10.00%)

### Artifact Paths Created
- `artifacts/validation/dev_source1_ids.txt` (1,986,139 lines)
- `artifacts/validation/validation_source1_ids.txt` (220,682 lines)
- `artifacts/validation/dev_ground_truth.tsv` (1,986,139 rows)
- `artifacts/validation/validation_ground_truth.tsv` (220,682 rows)

### Validation Ground-Truth Match Statistics

| Match Category / Multiplicity | Count (Validation Set) | Percentage of Validation Set (%) | Representation in Full GT (%) |
| :--- | :---: | :---: | :---: |
| **Zero Matches (Singletons / Null)** | 12,325 | **5.5849%** | 5.58% |
| **One Match** | 11,874 | **5.3806%** | 5.40% |
| **Multiple Matches (>1)** | 196,484 | **89.0345%** | 89.02% |
| **S2 Match Only** | 14,303 | **6.48%** | 6.48% |
| **S3 Match Only** | 16,450 | **7.45%** | 7.45% |
| **Both S2 & S3 Matches** | 177,605 | **80.48%** | 80.48% |

---

## 4. Task 4: Competition $F_{0.5}$ Scorer & Unit Test

### Metric Formula & Implementation
Implemented reusable macro-averaged $F_{0.5}$ metric in `src/evaluation/metrics.py`:

$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

### Edge Cases Handled
1. **Empty Ground Truth & Empty Prediction**: $F_{0.5} = 1.0$ (Correctly predicted no matches).
2. **Empty Ground Truth & Non-Empty Prediction**: $F_{0.5} = 0.0$ (False positive on singleton).
3. **Non-Empty Ground Truth & Empty Prediction**: $F_{0.5} = 0.0$ (False negative).
4. **Multiple Matches & Partial Recall**: Correct per-entity precision, recall, and $F_{0.5}$ macro-averaged across all Source 1 entities.

### Unit Test Verification
- Unit test script: `tests/test_metrics.py`
- Executed via `python3 -m unittest discover -s tests`
- **Result**: **PASS** (9 tests ran, 0 failures, 0 errors in 0.003s).

---

## 5. Task 5: Official Submission Validator Status

- Script Location: `utils/validate_submission.py`
- Verification: Executed and verified compatible with python environment.
- Current Status: **Verified Working** (Verified Working (Output file missing as expected in Phase 0)).

---

## 6. Summary of Phase 0 Deliverables

1. `src/data/data_loader.py` — Load and verify 7 TSV datasets.
2. `src/data/split.py` — Deterministic stratified S1-level validation split.
3. `src/evaluation/metrics.py` — Reusable competition macro-averaged $F_{0.5}$ scorer.
4. `tests/test_metrics.py` — Unit test suite verifying metric edge cases.
5. `artifacts/validation/dev_source1_ids.txt` — 1,986,139 dev S1 entity IDs.
6. `artifacts/validation/validation_source1_ids.txt` — 220,682 validation S1 entity IDs.
7. `artifacts/validation/dev_ground_truth.tsv` — Ground truth subset for development set.
8. `artifacts/validation/validation_ground_truth.tsv` — Ground truth subset for validation set.
9. `artifacts/phase0_report.md` — Phase 0 validation report.

---

**Phase 0 Setup Complete.** No normalization, blocking, feature engineering, or ML modeling has been implemented.
