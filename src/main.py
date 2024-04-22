import csv
import pandas as pd
import subprocess
from tqdm import tqdm

def process_csv(csv_file):
    
    seq_table = pd.read_csv(csv_file)
    left_scaffolds = {}
    for row in range(seq_table.shape[0]):
        id = seq_table.iloc[row, 0]
        sequence = ''.join(seq_table.iloc[row, 1])
        scaffold_left = sequence.replace(".", "")
        left_scaffolds[id] = scaffold_left
            
    return left_scaffolds

if __name__ == "__main__":
    csv_file_name = "data/TRBV_human_imgt.csv"
    num_repeats = 1

    left_scaffolds = process_csv(csv_file_name)
    with open('data/trbc2_seq.txt', 'r') as file:
        right_scaffold = file.read().strip()
    model_type = "oa_dm_38M"

    for id in tqdm(left_scaffolds.keys()):
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

