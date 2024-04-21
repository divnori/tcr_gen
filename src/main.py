import csv
import pandas as pd

def process_csv(csv_file):
    scaffold_regions = []
    cdr1_lengths = []
    cdr2_lengths = []
    cdr3_lengths = []
    
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        all_scaffolds = []
        all_cdr_lens = []
        for row in reader:
            sequence = ''.join(row)
            scaffolds = []
            cdr_lens = []

            curr_scaffold = True
            region_start = 0
            for i in range(len(sequence)):
                if sequence[i] == "." and curr_scaffold:
                    # to skip mask tokens
                    if (sequence[i-1] != "." and sequence[i+1] != ".") or len(scaffolds) == 3:
                        continue
                    scaffolds.append(sequence[region_start:i])
                    curr_scaffold = False
                    region_start = i
                elif sequence[i] != "." and not curr_scaffold:
                    cdr_lens.append(len(sequence[region_start:i]))
                    curr_scaffold = True
                    region_start = i

            print(scaffolds, cdr_lens)

            all_scaffolds.append(scaffolds)
            all_cdr_lens.append(cdr_lens)
    
    return all_scaffolds, all_cdr_lens

if __name__ == "__main__":

    all_scaffolds, all_cdr_lens = process_csv("TRAV_human_imgt.csv")