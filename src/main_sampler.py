import csv
import pandas as pd
import subprocess
from tqdm import tqdm

def process_V_regions(csv_file):
    
    seq_table = pd.read_csv(csv_file)
    left_scaffolds = {}
    for row in range(seq_table.shape[0]):
        id = seq_table.iloc[row, 0]
        sequence = ''.join(seq_table.iloc[row, 1])
        scaffold_left = sequence.replace(".", "")
        left_scaffolds[id] = scaffold_left
            
    return left_scaffolds

def process_J_regions(csv_file):
    
    seq_table = pd.read_csv(csv_file)
    mid_scaffolds = {}
    for row in range(seq_table.shape[0]):
        id = seq_table.iloc[row, 0]
        sequence = ''.join(seq_table.iloc[row, 1])
        scaffold_mid = sequence.replace(".", "")
        mid_scaffolds[id] = scaffold_mid
            
    return mid_scaffolds

def gen_uniform_VJ_table(V_list, J_list):
    f

if __name__ == "__main__":
    V_region_file = "data/TRJ_human_imgt.csv"
    J_region_file = "data/TRBV_human_imgt.csv"

    left_scaffolds = process_V_regions(V_region_file)
    mid_scaffolds = process_J_regions(J_region_file)
    with open('data/trbc2_seq.txt', 'r') as file:
        right_scaffold = file.read().strip()

    num_samples = 10000
    model_type = "oa_dm_38M"

    for i in range(num_samples):
        for cdr3_len in tqdm(range(8,12)): # first 3 residues already in prompt
            for j in range(num_repeats):
                command = [
                    "python",
                    "evodiff/evodiff/conditional_generation.py",
                    f"--model-type={model_type}",
                    "--gpus=6",
                    "--cond-task=tcr",
                    f"--scaffold_chain_left={left_scaffolds[id]}",
                    f"--scaffold_chain_right={right_scaffold}",
                    f"--cdr3_len={cdr3_len}"
                ]
                subprocess.run(command)