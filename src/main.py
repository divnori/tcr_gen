import csv
import pandas as pd
import subprocess

def process_csv(csv_file):
    scaffold_regions = []
    cdr1_lengths = []
    cdr2_lengths = []
    cdr3_lengths = []
    
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        left_scaffolds = []
        right_scaffolds = []
        for row in reader:
            sequence = ''.join(row)
            scaffold_left = sequence.replace(".", "")
            
    
    return left_scaffolds, right_scaffolds

if __name__ == "__main__":
    csv_file_name = "TRAV_human_imgt.csv"
    num_repeats = 1

    left_scaffolds, right_scaffolds = process_csv(csv_file_name)
    model_type = "oa_dm_38M"

    for i in range(len(all_scaffolds)):
        scaffold_chains = all_scaffolds[i]
        for cdr3_len in range(8,16):
            for j in range(num_repeats):
                command = [
                    "python",
                    "evodiff/evodiff/conditional_generation.py",
                    f"--model-type={model_type}",
                    "--gpus=6",
                    "--cond-task=tcr",
                    f"--scaffold_chain_left={left_scaffolds[i]}",
                    f"--scaffold_chain_right={right_scaffolds[i]}",
                    f"--cdr3_len={cdr3_len}"
                ]
                subprocess.run(command)
