import pandas as pd
import numpy as np
from statsmodels.stats import inter_rater as ir

def calculate_kappa_statsmodels(merged_df, col_base_name, annotator_suffixes):
    """
    Formats the data into a contingency table and calculates Fleiss' Kappa 
    using the statsmodels library.
    """
    # Select the columns for this specific task (e.g., subjectivity_detection_k, etc.)
    cols = [f"{col_base_name}_{s}" for s in annotator_suffixes]
    
    # Drop rows where any annotator left a blank
    data = merged_df[cols].dropna()
    
    # Get all unique labels present in these columns (e.g., 'neutral', 'positive')
    unique_categories = pd.unique(data.values.ravel())
    unique_categories = [cat for cat in unique_categories if pd.notnull(cat)]
    
    # Create the count matrix
    # Each row is a post, each column is a category, value is how many people picked it
    count_matrix = []
    for _, row in data.iterrows():
        row_list = list(row)
        counts = [row_list.count(cat) for cat in unique_categories]
        count_matrix.append(counts)
    
    # Calculate Fleiss Kappa
    kappa = ir.fleiss_kappa(np.array(count_matrix))
    return kappa, len(data)

# Load the Excel Files (change file paths to match the 3 different excel files accordingly) 
try:
    df_kavya = pd.read_excel('final_labels/kavya.xlsx')
    df_potala = pd.read_excel('final_labels/potala.xlsx')
    df_wallace = pd.read_excel('final_labels/wallace.xlsx')
except FileNotFoundError as e:
    print(f"Error: Could not find the file. {e}")
    exit()

# Prepare and Merge the Data
target_cols = ['post_id', 'subjectivity_detection', 'polarity_detection', 'sarcasm_detection']

# Rename columns to distinguish between annotators before merging
d1 = df_kavya[target_cols].rename(columns={c: c + '_k' for c in target_cols if c != 'post_id'})
d2 = df_potala[target_cols].rename(columns={c: c + '_p' for c in target_cols if c != 'post_id'})
d3 = df_wallace[target_cols].rename(columns={c: c + '_w' for c in target_cols if c != 'post_id'})

# Merge all three on the common post_id
merged = d1.merge(d2, on='post_id').merge(d3, on='post_id')

# Run Calculations
tasks = ['subjectivity_detection', 'polarity_detection', 'sarcasm_detection']
suffixes = ['k', 'p', 'w']

print(f"{'Detection Task':<25} | {'Kappa Score':<12} | {'Sample Size'}")
print("-" * 55)

for task in tasks:
    kappa_val, sample_n = calculate_kappa_statsmodels(merged, task, suffixes)
    print(f"{task:<25} | {kappa_val:<12.4f} | {sample_n}")

# Interpretation Guidelines
print("\nInterpretation Reference:")
print(" < 0.20: Slight Agreement")
print(" 0.21 - 0.40: Fair Agreement")
print(" 0.41 - 0.60: Moderate Agreement")
print(" 0.61 - 0.80: Substantial Agreement")