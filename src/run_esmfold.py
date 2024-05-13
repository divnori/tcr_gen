import biotite.structure.io as bsio
import csv
import esm
import os
import torch
from tqdm import tqdm

model = esm.pretrained.esmfold_v1()
model = model.eval().cuda()

# Path to the input CSV file
csv_file = 'results/trbv_gen.csv'

# Path to the results folder
results_folder = 'results/structures'

results_csv_path = 'results/plddt_results.csv'

with open(csv_file, 'r') as file:
    reader = csv.reader(file)
    sequences = [row[0] for row in reader]

with open(results_csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Sequence', 'pLDDT'])  
    
    for i, sequence in tqdm(enumerate(sequences)):

        try:
        
            with torch.no_grad():
                output = model.infer_pdb(sequence)

            with open(os.path.join(results_folder, f"prot{i}.pdb"), "w") as f:
                f.write(output)

            struct = bsio.load_structure(os.path.join(results_folder, f"prot{i}.pdb"), extra_fields=["b_factor"])
            plddt = struct.b_factor.mean()
            
            writer.writerow([sequence, plddt])

        except Exception as e:

            print(e)

            print("error on sequence", i)