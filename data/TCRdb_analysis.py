import numpy as np
import pandas as pd

dir = '/Users/ashyamal/Downloads/tcr_gen/data/'
df = pd.read_csv(dir + 'PRJNA390125.tsv', sep='\t', header=0)

# Check whether cloneFraction column sums to 1
#print(df['cloneFraction'].sum())
# It sums to 24 (# of ppl)

unique_vregions = df['Vregion'].unique()
unique_jregions = df['Jregion'].unique()

matrix = pd.DataFrame(0, index=unique_vregions, columns=unique_jregions)

for _, row in df.iterrows():
    matrix.loc[row['Vregion'], row['Jregion']] += float(row['cloneFraction']) / 24.0

# Print V and J region co-frequency matrix to generate joint distributions
#print(matrix)
matrix.to_csv(dir + 'normalized_V_J_freqs.csv')

length_counts = pd.DataFrame(columns=['Length', 'Count'])

for _, row in df.iterrows():
    length = len(row['AASeq'])

    if length not in length_counts['Length'].tolist():
        length_counts = length_counts._append({'Length': length, 'Count': 0}, ignore_index=True)
    length_counts.loc[length_counts['Length'] == length, 'Count'] += float(row['cloneFraction']) / 24.0

length_counts.sort_values(by='Length', inplace=True)
length_counts.reset_index(drop=True, inplace=True)

#print(length_counts)
length_counts.to_csv(dir + 'CDR_length_freqs.csv', index=False)