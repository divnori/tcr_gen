import pandas as pd

df = pd.read_csv('PRJNA390125.tsv', sep='\t', header=0)

unique_vregions = df['Vregion'].unique()
unique_jregions = df['Jregion'].unique()

matrix = pd.DataFrame(0, index=unique_vregions, columns=unique_jregions)

for _, row in df.iterrows():
    matrix.loc[row['Vregion'], row['Jregion']] += 1

# Check whether cloneFraction colun sums to 1
print(df['cloneFraction'].sum())

# Print V and J region co-frequency matrix to generate joint distributions
print(matrix)