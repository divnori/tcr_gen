import csv
import pandas as pd
import subprocess
from tqdm import tqdm
import numpy as np
import pandas as pd
import random

def process_scaffolds(csv_file):
    seq_table = pd.read_csv(csv_file)
    scaffolds = []
    scaffold_ids = []
    for row in range(seq_table.shape[0]):
        id = seq_table.iloc[row, 0]
        sequence = ''.join(seq_table.iloc[row, 1])
        scaffolds.append(sequence.replace(".", ""))
        scaffold_ids.append(id)
    return pd.Series(scaffolds, index=scaffold_ids)

def gen_uniform_VJ_table(V_list, J_list):
    """
    Creates a VJ sampler assuming uniform usage across all VJ pairs (just for testing, not realistic!)
    """
    freq = 1/(len(V_list) * len(J_list))
    VJ_ids = [f'{V};{J}' for V in V_list for J in J_list]
    return pd.Series([freq for i in range(len(VJ_ids))], index=VJ_ids)

def gen_uniform_CDR3_table(CDR3_lengths):
    freq = 1/len(CDR3_lengths)
    return pd.Series([freq for i in range(len(CDR3_lengths))], index=CDR3_lengths)

def load_sequence_VJ_table(csv_file):
    VJ_mat = pd.read_csv(csv_file, index_col='V_region')
    VJ_freqs = []
    VJ_ids = []
    for r in range(VJ_mat.shape[0]):
        for c in range(VJ_mat.shape[1]):
            VJ_freqs.append(VJ_mat.iloc[r,c])
            VJ_ids.append(f'{VJ_mat.index[r]};{VJ_mat.columns[c]}')
    return pd.Series(VJ_freqs, index=VJ_ids)


def sample_VJ(k, V_scaffolds, J_scaffolds, VJ_table):
    """
    Creates a CDR3 length sampler assuming uniform length distribution (just for testing, not realistic!)
    """
    for choice in random.choices(k=k, population=VJ_table.index, weights=VJ_table.values):
        V_id, J_id = choice.split(';')
        print(V_id, J_id)
        yield V_scaffolds[V_id], J_scaffolds[J_id]

def sample_CDR3(k, CDR3_table):
    for choice in random.choices(k=k, population=CDR3_table.index, weights=CDR3_table.values):
        yield choice

if __name__ == "__main__":
    chain = 'B'
    V_region_file = f"data/TR{chain}V_human_imgt.csv"
    J_region_file = f"data/TR{chain}J_human_imgt.csv"

    V_scaffolds = process_scaffolds(V_region_file)
    J_scaffolds = process_scaffolds(J_region_file)
    with open(f'data/TR{chain}C_human_imgt.txt', 'r') as file:
        C_scaffold = file.read().strip()

    VJ_table = load_sequence_VJ_table('data/normalized_V_J_freqs.csv')
    CDR3_table = gen_uniform_CDR3_table(np.arange(8, 22))
    
    num_samples = 1000
    model_type = "oa_dm_38M"
    VJ_gen = sample_VJ(num_samples, V_scaffolds, J_scaffolds, VJ_table)
    CDR3_gen = sample_CDR3(num_samples, CDR3_table)

    for i in tqdm(range(num_samples)):
        V_scaffold, J_scaffold = next(VJ_gen)
        CDR3_len = next(CDR3_gen)
        left_scaffold = V_scaffold
        right_scaffold = J_scaffold + C_scaffold
        # print(V_scaffold)
        # print(J_scaffold)
        # print(CDR3_len)
        # print()

        command = [
            "python",
            "evodiff/evodiff/conditional_generation.py",
            f"--model-type={model_type}",
            "--gpus=6",
            "--cond-task=tcr",
            f"--scaffold_chain_left={left_scaffold}",
            f"--scaffold_chain_right={right_scaffold}",
            f"--cdr3_len={CDR3_len}"
        ]
        subprocess.run(command)
