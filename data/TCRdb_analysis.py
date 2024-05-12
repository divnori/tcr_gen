import numpy as np
import pandas as pd

dir = '/Users/ashyamal/Downloads/tcr_gen/data/'
df = pd.read_csv(dir + 'PRJNA390125.tsv', sep='\t', header=0)

unique_vregions = df['Vregion'].unique()
unique_jregions = df['Jregion'].unique()

matrix = pd.DataFrame(0, index=unique_vregions, columns=unique_jregions)

for _, row in df.iterrows():
    matrix.loc[row['Vregion'], row['Jregion']] += float(row['cloneFraction']) / 24.0

# Check whether cloneFraction colun sums to 1
#print(df['cloneFraction'].sum())
# It sums to 24 (# of ppl)

# Print V and J region co-frequency matrix to generate joint distributions
print(matrix)
matrix.to_csv(dir + 'normalized_V_J_freqs.csv')
