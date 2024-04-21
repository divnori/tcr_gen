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
        all_scaffolds = []
        all_cdr_placeholders = []
        for row in reader:
            sequence = ''.join(row)
            scaffolds = []
            cdr_placeholders = []

            curr_scaffold = True
            region_start = 0
            for i in range(len(sequence)):
                if sequence[i] == "." and curr_scaffold:
                    # to skip mask tokens
                    if (i < len(sequence) - 1 and sequence[i-1] != "." and sequence[i+1] != ".") or len(scaffolds) == 3:
                        continue
                    scaffolds.append(sequence[region_start:i].replace(".",""))
                    curr_scaffold = False
                    region_start = i
                elif i == len(sequence)-1:
                    scaffolds.append(sequence[region_start:i].replace(".",""))
                elif sequence[i] != "." and not curr_scaffold:
                    cdr_placeholders.append(len(sequence[region_start:i]))
                    curr_scaffold = True
                    region_start = i

            all_scaffolds.append(scaffolds)
            all_cdr_placeholders.append(cdr_placeholders)
    
    return all_scaffolds, all_cdr_placeholders

if __name__ == "__main__":

    all_scaffolds, all_cdr_placeholders = process_csv("TRAV_human_imgt.csv")

    model_type = "oa_dm_38M"
    cdr1_len = 8
    cdr2_len = 8

    for i in range(len(all_scaffolds)):
        scaffold_chains = all_scaffolds[i]
        print(scaffold_chains)

        for cdr3_len in range(8,16):

            command = [
                "python",
                "evodiff/evodiff/conditional_generation.py",
                f"--model-type={model_type}",
                "--gpus=1",
                "--cond-task=tcr",
                f"--scaffold_chain1={scaffold_chains[0]}",
                f"--scaffold_chain2={scaffold_chains[1]}",
                f"--scaffold_chain3={scaffold_chains[2]}",
                f"--scaffold_chain4={scaffold_chains[3]}",
                f"--cdr1_len={cdr1_len}",
                f"--cdr2_len={cdr2_len}",
                f"--cdr3_len={cdr3_len}"
            ]

            subprocess.run(command)