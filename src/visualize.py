import biotite.structure.io as bsio
import matplotlib.pyplot as plt
import os
import pandas as pd

def make_global_plddt_histogram(results_csv_path):
    df = pd.read_csv(results_csv_path)
    mean_plddt = df['pLDDT'].mean()
    print(f'Mean pLDDT: {mean_plddt}')
    plt.hist(df['pLDDT'], bins=20, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('pLDDT')
    plt.ylabel('Frequency')
    plt.title('pLDDT Histogram')
    plt.savefig('results/plddt_histogram.png', dpi=200)

def make_per_protein_plddt_plot(structures_path, prot_idx):
    pdb_path = os.path.join(structures_path, f'prot{prot_idx}.pdb')
    struct = bsio.load_structure(pdb_path, extra_fields=["b_factor"])
    plddt = struct.b_factor.mean()
    print(f'pLDDT: {plddt}')

    cdr_df = pd.read_csv('results/trbv_gen.csv')
    left_context = cdr_df['left_context'].values[prot_idx]
    right_context = cdr_df['right_context'].values[prot_idx]
    full_sequence = cdr_df['full_sequence'].values[prot_idx]
    generated_cdr = full_sequence[len(left_context):len(full_sequence)-len(right_context)]
    plt.plot(range(len(struct.b_factor)), struct.b_factor, color='blue', label='pLDDT')
    plt.axvspan(len(left_context), len(left_context)+len(generated_cdr), color='yellow', alpha=0.3, label='Generated CDR')

    plt.xlabel('Sequence Position')
    plt.ylabel('pLDDT')
    plt.title('pLDDT by Sequence Position')
    plt.legend()
    plt.savefig('results/plddt_by_position.png', dpi=200)


if __name__ == "__main__":

    # make_global_plddt_histogram('results/plddt_results.csv')

    make_per_protein_plddt_plot('results/structures', 0)