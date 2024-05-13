import csv
import pandas as pd

def load_data():
    print("Loading data...")
    
    trb_gt = pd.read_csv("data/TRB_CDR3_human_VDJdb.tsv",sep='\t')
    trb_gt = trb_gt[trb_gt["Gene"] == "TRBV"]
    with open('/home/dnori/tcr_gen/data/TRBV_human_imgt.csv', newline='') as f:
        reader = csv.DictReader(f)
        v_map = {row.pop('id'): row.pop('sequence') for row in reader}

    cdr3s = trb_gt["CDR3"].tolist()
    v_ids = trb_gt["V"].tolist()
    v_regions = [v_map[v_ids[i].split("*")[0]] for i in range(len(v_ids))]

    return v_regions, cdr3s

if __name__ == "__main__":

    v_regions, cdr3s = load_data()