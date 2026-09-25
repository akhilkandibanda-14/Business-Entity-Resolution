import os
import re
import math
import json
import glob
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import nbformat as nbf

# Set style for charts
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

# Setup output directories
os.makedirs('eda/figures', exist_ok=True)
os.makedirs('eda/tables', exist_ok=True)

print("=== STARTING FAST OPTIMIZED EDA DATA ENGINE ===", flush=True)
t0 = time.time()

# -------------------------------------------------------------
# 1. Dataset Files Manifest
# -------------------------------------------------------------
train_files = {
    'train_s1': 'dataset/train/train_source1.tsv',
    'train_s2': 'dataset/train/train_source2.tsv',
    'train_s3': 'dataset/train/train_source3.tsv',
    'train_gt': 'dataset/train/train_ground_truth.tsv'
}

test_files = {
    'test_s1': 'dataset/test/test_source1.tsv',
    'test_s2': 'dataset/test/test_source2.tsv',
    'test_s3': 'dataset/test/test_source3.tsv'
}

all_files = {**train_files, **test_files}

# -------------------------------------------------------------
# 2. Basic Dataset Metrics
# -------------------------------------------------------------
print("\n[Step 1/6] Running Basic Dataset Analysis...", flush=True)
basic_metrics = []

for name, path in all_files.items():
    df = pd.read_csv(path, sep='\t')
    num_rows, num_cols = df.shape
    cols = list(df.columns)
    first_col = cols[0]
    dtypes = {c: str(df[c].dtype) for c in df.columns}
    null_counts = df.isnull().sum().to_dict()
    null_pcts = (df.isnull().sum() / num_rows * 100).round(4).to_dict()
    dup_rows = int(df.duplicated().sum())
    unique_ids = int(df[first_col].nunique())
    dup_ids = num_rows - unique_ids
    
    basic_metrics.append({
        'File Key': name,
        'File Path': path,
        'Rows': num_rows,
        'Cols': num_cols,
        'Columns': ", ".join(cols),
        'Data Types': json.dumps(dtypes),
        'Duplicate Rows': dup_rows,
        f'Unique {first_col}': unique_ids,
        'Duplicate Entity IDs': dup_ids,
        'Null Counts': json.dumps(null_counts),
        'Null Percentages': json.dumps(null_pcts)
    })

df_basic = pd.DataFrame(basic_metrics)
df_basic.to_csv('eda/tables/basic_dataset_analysis.csv', index=False)
print("Saved basic dataset metrics to eda/tables/basic_dataset_analysis.csv", flush=True)

# -------------------------------------------------------------
# 3. Country Analysis
# -------------------------------------------------------------
print("\n[Step 2/6] Running Country Distribution Analysis...", flush=True)
country_summary = []
country_pivot = {}

for name, path in all_files.items():
    if name == 'train_gt':
        continue
    df = pd.read_csv(path, sep='\t', usecols=['country'])
    total = len(df)
    counts = df['country'].value_counts().to_dict()
    pcts = (df['country'].value_counts(normalize=True)*100).round(4).to_dict()
    country_pivot[name] = pcts
    
    for c_code, count in counts.items():
        country_summary.append({
            'Source': name,
            'Country': c_code,
            'Count': count,
            'Percentage': pcts[c_code]
        })

df_country = pd.DataFrame(country_summary)
df_country.to_csv('eda/tables/country_analysis.csv', index=False)

# Plot Country Distribution
fig, ax = plt.subplots(figsize=(10, 5))
df_pivot = pd.DataFrame(country_pivot).fillna(0)
df_pivot.T.plot(kind='bar', stacked=False, ax=ax, colormap='viridis', edgecolor='black', alpha=0.85)
ax.set_title('Country Distribution Across All Data Sources', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Dataset Source', fontsize=12, labelpad=10)
ax.set_ylabel('Percentage (%)', fontsize=12)
ax.set_ylim(0, 70)
plt.xticks(rotation=0)
for p in ax.patches:
    h = p.get_height()
    if h > 0:
        ax.annotate(f'{h:.1f}%', (p.get_x() + p.get_width() / 2., h + 1),
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
plt.tight_layout()
plt.savefig('eda/figures/country_distribution_comparison.png', dpi=300)
plt.close()
print("Saved country distribution chart to eda/figures/country_distribution_comparison.png", flush=True)

# -------------------------------------------------------------
# 4. Ground Truth & Match Type Analysis
# -------------------------------------------------------------
print("\n[Step 3/6] Analyzing Ground Truth & Match Types...", flush=True)
gt = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t')
total_s1 = len(gt)

null_mask = gt['matched_entity_ids'].isnull()
no_match_cnt = int(null_mask.sum())

valid_gt = gt[~null_mask].copy()
valid_gt['match_list'] = valid_gt['matched_entity_ids'].str.split(',')
valid_gt['num_matches'] = valid_gt['match_list'].apply(len)

valid_gt['s2_matches'] = valid_gt['match_list'].apply(lambda lst: [x for x in lst if x.startswith('S2-')])
valid_gt['s3_matches'] = valid_gt['match_list'].apply(lambda lst: [x for x in lst if x.startswith('S3-')])

valid_gt['num_s2'] = valid_gt['s2_matches'].apply(len)
valid_gt['num_s3'] = valid_gt['s3_matches'].apply(len)

s1_s2_only = int(((valid_gt['num_s2'] > 0) & (valid_gt['num_s3'] == 0)).sum())
s1_s3_only = int(((valid_gt['num_s2'] == 0) & (valid_gt['num_s3'] > 0)).sum())
s1_both = int(((valid_gt['num_s2'] > 0) & (valid_gt['num_s3'] > 0)).sum())

total_s2_links = int(valid_gt['num_s2'].sum())
total_s3_links = int(valid_gt['num_s3'].sum())

# Match count distribution
match_counts = pd.Series(0, index=gt.index)
match_counts.loc[valid_gt.index] = valid_gt['num_matches']

match_dist_counts = match_counts.value_counts().sort_index().to_dict()

gt_stats = {
    'Total S1 Entities': total_s1,
    'S1 Zero Matches': no_match_cnt,
    'S1 Zero Matches Pct': round(no_match_cnt / total_s1 * 100, 4),
    'S1 S2 Only Matches': s1_s2_only,
    'S1 S2 Only Pct': round(s1_s2_only / total_s1 * 100, 4),
    'S1 S3 Only Matches': s1_s3_only,
    'S1 S3 Only Pct': round(s1_s3_only / total_s1 * 100, 4),
    'S1 Both S2 & S3 Matches': s1_both,
    'S1 Both Pct': round(s1_both / total_s1 * 100, 4),
    'Total S1-S2 Matching Links': total_s2_links,
    'Total S1-S3 Matching Links': total_s3_links,
    'Total Matching Links': total_s2_links + total_s3_links,
    'Mean Matches per S1': round(float(match_counts.mean()), 4),
    'Median Matches per S1': float(match_counts.median()),
    'Max Matches per S1': int(match_counts.max()),
    'Match Count Dist': json.dumps(match_dist_counts)
}

pd.DataFrame([gt_stats]).to_csv('eda/tables/ground_truth_analysis.csv', index=False)

# Ground Truth Visualizations
fig, ax = plt.subplots(figsize=(9, 5))
counts_series = pd.Series(match_dist_counts)
bars = ax.bar(counts_series.index, counts_series.values, color='#2b5c8f', edgecolor='black', alpha=0.85)
ax.set_title('Distribution of Ground-Truth Match Counts per Source 1 Entity', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Number of Matched Target Entities (S2 + S3)', fontsize=12)
ax.set_ylabel('Source 1 Entity Count', fontsize=12)
ax.set_xticks(range(0, 12))
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 10000, f'{yval:,}', ha='center', va='bottom', fontsize=8, rotation=30)
plt.tight_layout()
plt.savefig('eda/figures/ground_truth_matches_dist.png', dpi=300)
plt.close()

fig, ax = plt.subplots(figsize=(7, 7))
labels = [
    f'Both S2 & S3\n({s1_both:,} | {s1_both/total_s1*100:.1f}%)',
    f'S3 Only\n({s1_s3_only:,} | {s1_s3_only/total_s1*100:.1f}%)',
    f'S2 Only\n({s1_s2_only:,} | {s1_s2_only/total_s1*100:.1f}%)',
    f'No Matches\n({no_match_cnt:,} | {no_match_cnt/total_s1*100:.1f}%)'
]
sizes = [s1_both, s1_s3_only, s1_s2_only, no_match_cnt]
colors = ['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728']
ax.pie(sizes, labels=labels, colors=colors, autopct='', startangle=140, wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2))
ax.set_title('Source 1 Match Category Distribution', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('eda/figures/match_types_pie.png', dpi=300)
plt.close()
print("Saved ground truth metrics and charts.", flush=True)

# -------------------------------------------------------------
# 5. Name and Address Analysis across all sources (Sampled for token percentages)
# -------------------------------------------------------------
print("\n[Step 4/6] Running Fast Business Name & Address Feature Extraction...", flush=True)

legal_pattern = r'\b(?:ltd|limited|pvt|private|inc|incorporated|corp|corporation|llc|co|company|gmbh|sarl|sas|sa|nv|bv|pty|plc|spa|s\.p\.a\.|s\.a\.|eurl|llp)\b'

addr_patterns = {
    'street_st': r'\b(?:street|st)\b',
    'road_rd': r'\b(?:road|rd)\b',
    'avenue_ave': r'\b(?:avenue|ave)\b',
    'suite_ste': r'\b(?:suite|ste)\b',
    'floor_fl': r'\b(?:floor|fl)\b',
    'po_box': r'\b(?:p\.?o\.?\s*box|post\s*office\s*box)\b',
    'landmark_near_opp': r'\b(?:near|opp|opposite|behind|beside)\b',
    'french_terms': r'\b(?:rue|avenue|boulevard|blvd|place|allée|chemin|route|impedance|cedex)\b'
}

name_metrics_list = []
addr_metrics_list = []

name_lens_dict = {}
addr_lens_dict = {}

for name, path in all_files.items():
    if name == 'train_gt':
        continue
    print(f"Processing source: {name}...", flush=True)
    df = pd.read_csv(path, sep='\t')
    total_rows = len(df)
    
    # --- NAME ANALYSIS ---
    names = df['business_name'].dropna().astype(str)
    missing_names = int(df['business_name'].isnull().sum())
    unique_names = int(names.nunique())
    dup_names = total_rows - unique_names
    
    char_lens = names.str.len()
    word_counts = names.str.split().str.len()
    
    # Sample for regex & case stats to run fast
    sample_size = min(300000, len(names))
    sample_names = names.sample(n=sample_size, random_state=42)
    
    is_upper_pct = (sample_names.str.isupper().sum() / sample_size * 100)
    is_title_pct = (sample_names.str.istitle().sum() / sample_size * 100)
    has_legal_pct = (sample_names.str.contains(legal_pattern, case=False, regex=True).sum() / sample_size * 100)
    
    name_lens_dict[name] = char_lens.values
    
    very_short_names = int((char_lens < 3).sum())
    very_long_names = int((char_lens > 100).sum())
    
    name_metrics_list.append({
        'Source': name,
        'Total Rows': total_rows,
        'Missing Names': missing_names,
        'Missing Pct': round(missing_names / total_rows * 100, 4),
        'Unique Names': unique_names,
        'Duplicate Names': dup_names,
        'Mean Char Len': round(float(char_lens.mean()), 2),
        'Median Char Len': float(char_lens.median()),
        'Max Char Len': int(char_lens.max()),
        'Min Char Len': int(char_lens.min()),
        'Mean Word Count': round(float(word_counts.mean()), 2),
        'Median Word Count': float(word_counts.median()),
        'ALL CAPS Pct': round(is_upper_pct, 2),
        'Title Case Pct': round(is_title_pct, 2),
        'Has Legal Suffix Pct': round(has_legal_pct, 2),
        'Very Short (<3 chars)': very_short_names,
        'Very Long (>100 chars)': very_long_names
    })
    
    # --- ADDRESS ANALYSIS ---
    addrs = df['business_address'].dropna().astype(str)
    missing_addrs = int(df['business_address'].isnull().sum())
    unique_addrs = int(addrs.nunique())
    dup_addrs = total_rows - unique_addrs
    
    addr_char_lens = addrs.str.len()
    addr_word_counts = addrs.str.split().str.len()
    
    addr_lens_dict[name] = addr_char_lens.values
    
    sample_addr_size = min(300000, len(addrs))
    sample_addrs = addrs.sample(n=sample_addr_size, random_state=42)
    
    has_postcode_pct = (sample_addrs.str.contains(r'\b\d{5,6}\b', regex=True).sum() / sample_addr_size * 100)
    
    addr_tok_counts = {}
    for tok_k, tok_regex in addr_patterns.items():
        cnt = sample_addrs.str.contains(tok_regex, case=False, regex=True).sum()
        addr_tok_counts[f'Addr_{tok_k}_Pct'] = round(cnt / sample_addr_size * 100, 2)
        
    addr_metrics_list.append({
        'Source': name,
        'Total Rows': total_rows,
        'Missing Address': missing_addrs,
        'Missing Address Pct': round(missing_addrs / total_rows * 100, 4),
        'Unique Addresses': unique_addrs,
        'Duplicate Addresses': dup_addrs,
        'Mean Addr Char Len': round(float(addr_char_lens.mean()), 2),
        'Median Addr Char Len': float(addr_char_lens.median()),
        'Max Addr Char Len': int(addr_char_lens.max()),
        'Min Addr Char Len': int(addr_char_lens.min()),
        'Mean Addr Word Count': round(float(addr_word_counts.mean()), 2),
        'Has 5/6-Digit Postcode Pct': round(has_postcode_pct, 2),
        **addr_tok_counts
    })

pd.DataFrame(name_metrics_list).to_csv('eda/tables/business_name_analysis.csv', index=False)
pd.DataFrame(addr_metrics_list).to_csv('eda/tables/business_address_analysis.csv', index=False)

# Boxplot of Name Lengths
fig, ax = plt.subplots(figsize=(10, 5))
name_lens_df = pd.DataFrame({k: pd.Series(v[:50000]) for k, v in name_lens_dict.items()})
sns.boxplot(data=name_lens_df, ax=ax, palette='Blues')
ax.set_title('Business Name Character Length Distribution Across Sources', fontsize=14, fontweight='bold', pad=15)
ax.set_ylabel('Character Length', fontsize=12)
ax.set_ylim(0, 120)
plt.tight_layout()
plt.savefig('eda/figures/name_length_distribution.png', dpi=300)
plt.close()

# Boxplot of Address Lengths
fig, ax = plt.subplots(figsize=(10, 5))
addr_lens_df = pd.DataFrame({k: pd.Series(v[:50000]) for k, v in addr_lens_dict.items()})
sns.boxplot(data=addr_lens_df, ax=ax, palette='Oranges')
ax.set_title('Business Address Character Length Distribution Across Sources', fontsize=14, fontweight='bold', pad=15)
ax.set_ylabel('Character Length', fontsize=12)
ax.set_ylim(0, 200)
plt.tight_layout()
plt.savefig('eda/figures/address_length_distribution.png', dpi=300)
plt.close()

print("Saved Name & Address Analysis tables and figures.", flush=True)

# -------------------------------------------------------------
# 6. Match Pattern & Pairwise Similarity Analysis (Ground Truth)
# -------------------------------------------------------------
print("\n[Step 5/6] Performing Pairwise Match Pattern Analysis on True Ground Truth Links...", flush=True)

# Fast loading with specific columns
s1_df = pd.read_csv('dataset/train/train_source1.tsv', sep='\t', usecols=['entity_id', 'business_name', 'business_address', 'country']).set_index('entity_id')
s2_df = pd.read_csv('dataset/train/train_source2.tsv', sep='\t', usecols=['entity_id', 'business_name', 'business_address', 'country']).set_index('entity_id')
s3_df = pd.read_csv('dataset/train/train_source3.tsv', sep='\t', usecols=['entity_id', 'business_name', 'business_address', 'country']).set_index('entity_id')

valid_gt_sample = valid_gt.sample(n=25000, random_state=42)

matched_pair_records = []

def jaccard_similarity(str1, str2):
    if not isinstance(str1, str) or not isinstance(str2, str):
        return 0.0
    toks1 = set(re.findall(r'\w+', str1.lower()))
    toks2 = set(re.findall(r'\w+', str2.lower()))
    if not toks1 or not toks2:
        return 0.0
    return len(toks1 & toks2) / len(toks1 | toks2)

print("Sampling 25,000 ground-truth matching pairs and calculating metrics...", flush=True)
for _, row in valid_gt_sample.iterrows():
    s1_id = row['source1_entity_id']
    if s1_id not in s1_df.index:
        continue
    s1_row = s1_df.loc[s1_id]
    
    # Process S2 matches
    for s2_id in row['s2_matches'][:2]:
        if s2_id in s2_df.index:
            s2_row = s2_df.loc[s2_id]
            
            n1, n2 = str(s1_row['business_name']), str(s2_row['business_name'])
            a1, a2 = str(s1_row['business_address']), str(s2_row['business_address'])
            c1, c2 = str(s1_row['country']), str(s2_row['country'])
            
            name_exact = (n1 == n2)
            name_lower_exact = (n1.lower().strip() == n2.lower().strip())
            name_jaccard = jaccard_similarity(n1, n2)
            
            addr_exact = (a1 == a2)
            addr_jaccard = jaccard_similarity(a1, a2)
            country_match = (c1 == c2)
            
            matched_pair_records.append({
                'Pair Type': 'S1-S2',
                'S1 ID': s1_id,
                'Target ID': s2_id,
                'Country Match': country_match,
                'Name Exact': name_exact,
                'Name Case Insensitive Exact': name_lower_exact,
                'Name Token Jaccard': round(name_jaccard, 4),
                'Addr Exact': addr_exact,
                'Addr Token Jaccard': round(addr_jaccard, 4)
            })
            
    # Process S3 matches
    for s3_id in row['s3_matches'][:2]:
        if s3_id in s3_df.index:
            s3_row = s3_df.loc[s3_id]
            
            n1, n3 = str(s1_row['business_name']), str(s3_row['business_name'])
            a1, a3 = str(s1_row['business_address']), str(s3_row['business_address'])
            c1, c3 = str(s1_row['country']), str(s3_row['country'])
            
            name_exact = (n1 == n3)
            name_lower_exact = (n1.lower().strip() == n3.lower().strip())
            name_jaccard = jaccard_similarity(n1, n3)
            
            addr_exact = (a1 == a3)
            addr_jaccard = jaccard_similarity(a1, a3)
            country_match = (c1 == c3)
            
            matched_pair_records.append({
                'Pair Type': 'S1-S3',
                'S1 ID': s1_id,
                'Target ID': s3_id,
                'Country Match': country_match,
                'Name Exact': name_exact,
                'Name Case Insensitive Exact': name_lower_exact,
                'Name Token Jaccard': round(name_jaccard, 4),
                'Addr Exact': addr_exact,
                'Addr Token Jaccard': round(addr_jaccard, 4)
            })

df_pairs = pd.DataFrame(matched_pair_records)
df_pairs.to_csv('eda/tables/matched_pair_similarity_analysis.csv', index=False)

print("\n--- GROUND TRUTH MATCH PATTERN METRICS ---", flush=True)
print(f"Total Evaluated Matched Pairs: {len(df_pairs):,}", flush=True)
print(f"Country Agreement Rate: {df_pairs['Country Match'].mean()*100:.4f}%", flush=True)
print(f"Exact Name Match Rate: {df_pairs['Name Exact'].mean()*100:.2f}%", flush=True)
print(f"Case-Insensitive Exact Name Rate: {df_pairs['Name Case Insensitive Exact'].mean()*100:.2f}%", flush=True)
print(f"Mean Name Token Jaccard Similarity: {df_pairs['Name Token Jaccard'].mean():.4f}", flush=True)
print(f"Exact Address Match Rate: {df_pairs['Addr Exact'].mean()*100:.2f}%", flush=True)
print(f"Mean Address Token Jaccard Similarity: {df_pairs['Addr Token Jaccard'].mean():.4f}", flush=True)

# Visualizing Similarity Distributions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(df_pairs['Name Token Jaccard'], bins=20, ax=ax1, color='#2ca02c', kde=True)
ax1.set_title('Name Token Jaccard Similarity (True Matches)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Jaccard Similarity Score', fontsize=10)

sns.histplot(df_pairs['Addr Token Jaccard'], bins=20, ax=ax2, color='#ff7f0e', kde=True)
ax2.set_title('Address Token Jaccard Similarity (True Matches)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Jaccard Similarity Score', fontsize=10)

plt.tight_layout()
plt.savefig('eda/figures/matched_pair_similarity_dist.png', dpi=300)
plt.close()

# -------------------------------------------------------------
# 7. Generate Jupyter Notebook eda/entity_resolution_eda.ipynb
# -------------------------------------------------------------
print("\n[Step 6/6] Creating eda/entity_resolution_eda.ipynb...", flush=True)

nb = nbf.v4.new_notebook()
nb.cells = []

# Title cell
nb.cells.append(nbf.v4.new_markdown_cell("""# Business Entity Resolution ML Challenge - Comprehensive EDA

This Jupyter Notebook performs end-to-end Exploratory Data Analysis (EDA) across all 7 TSV files:
- `train_source1.tsv`, `train_source2.tsv`, `train_source3.tsv`, `train_ground_truth.tsv`
- `test_source1.tsv`, `test_source2.tsv`, `test_source3.tsv`

## Table of Contents
1. Environment Setup & Data Loading
2. Basic Dataset Statistics & Metadata
3. Business Name Analysis
4. Business Address Analysis
5. Country Distribution & Cross-Split Shift
6. Ground Truth Match Analysis
7. Match-Type Classification (S1->S2, S1->S3, Both)
8. Ground-Truth Match Pattern & Pairwise Similarity Analysis
9. Source Comparison Matrix & Summary Findings
"""))

# Cell 1: Imports
nb.cells.append(nbf.v4.new_code_cell("""import os
import re
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
print("Libraries loaded successfully.")
"""))

# Cell 2: Load Basic Dataset Summary Table
nb.cells.append(nbf.v4.new_markdown_cell("## 1. Basic Dataset Statistics"))
nb.cells.append(nbf.v4.new_code_cell("""df_basic = pd.read_csv('eda/tables/basic_dataset_analysis.csv')
df_basic[['File Key', 'Rows', 'Cols', 'Duplicate Rows', 'Duplicate Entity IDs']]
"""))

# Cell 3: Country Analysis
nb.cells.append(nbf.v4.new_markdown_cell("## 2. Country Distribution & Unseen Test Country"))
nb.cells.append(nbf.v4.new_code_cell("""df_country = pd.read_csv('eda/tables/country_analysis.csv')
print(df_country)

# Display saved country distribution chart
from IPython.display import Image
Image(filename='eda/figures/country_distribution_comparison.png')
"""))

# Cell 4: Ground Truth Analysis
nb.cells.append(nbf.v4.new_markdown_cell("## 3. Ground Truth & Match Types"))
nb.cells.append(nbf.v4.new_code_cell("""df_gt = pd.read_csv('eda/tables/ground_truth_analysis.csv')
for col in df_gt.columns:
    print(f"{col}: {df_gt[col].values[0]}")

Image(filename='eda/figures/ground_truth_matches_dist.png')
"""))

# Cell 5: Business Name Analysis
nb.cells.append(nbf.v4.new_markdown_cell("## 4. Business Name Feature Analysis"))
nb.cells.append(nbf.v4.new_code_cell("""df_name = pd.read_csv('eda/tables/business_name_analysis.csv')
df_name[['Source', 'Total Rows', 'Missing Names', 'Unique Names', 'Mean Char Len', 'ALL CAPS Pct', 'Has Legal Suffix Pct']]
"""))

# Cell 6: Business Address Analysis
nb.cells.append(nbf.v4.new_markdown_cell("## 5. Business Address Feature Analysis"))
nb.cells.append(nbf.v4.new_code_cell("""df_addr = pd.read_csv('eda/tables/business_address_analysis.csv')
df_addr[['Source', 'Total Rows', 'Missing Address Pct', 'Unique Addresses', 'Mean Addr Char Len', 'Has 5/6-Digit Postcode Pct']]
"""))

# Cell 7: Pairwise Match Pattern Analysis
nb.cells.append(nbf.v4.new_markdown_cell("## 6. Ground-Truth Match Pattern Analysis"))
nb.cells.append(nbf.v4.new_code_cell("""df_pairs = pd.read_csv('eda/tables/matched_pair_similarity_analysis.csv')
print(f"Total Evaluated True Pairs: {len(df_pairs):,}")
print(f"Country Agreement Rate: {df_pairs['Country Match'].mean()*100:.4f}%")
print(f"Exact Name Match Rate: {df_pairs['Name Exact'].mean()*100:.2f}%")
print(f"Exact Address Match Rate: {df_pairs['Addr Exact'].mean()*100:.2f}%")
print(f"Mean Name Token Jaccard: {df_pairs['Name Token Jaccard'].mean():.4f}")
print(f"Mean Address Token Jaccard: {df_pairs['Addr Token Jaccard'].mean():.4f}")

Image(filename='eda/figures/matched_pair_similarity_dist.png')
"""))

with open('eda/entity_resolution_eda.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Created eda/entity_resolution_eda.ipynb successfully.", flush=True)
print(f"\n=== FAST EDA DATA ENGINE COMPLETE IN {time.time()-t0:.2f} SECONDS ===", flush=True)
