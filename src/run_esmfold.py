import biotite.structure.io as bsio
import csv
import esm
import os
import torch

model = esm.pretrained.esmfold_v1()
model = model.eval().cuda()

# Path to the input CSV file
csv_file = 'results/trbv_gen_example.csv'

# Path to the results folder
results_folder = 'results/structures'

with open(csv_file, 'r') as file:
    reader = csv.reader(file)
    sequences = [row[0] for row in reader]

plddts = []
for i, sequence in enumerate(sequences):
    
    with torch.no_grad():
        output = model.infer_pdb(sequence)

    with open(os.path.join(results_folder, f"prot{i}.pdb"), "w") as f:
        f.write(output)

    struct = bsio.load_structure("result.pdb", extra_fields=["b_factor"])
    plddt = struct.b_factor.mean()
    plddts.append(plddt)

output_file = 'results/plddts.csv'
with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Sequence', 'PLDDT'])
    for sequence, plddt in zip(sequences, plddts):
        writer.writerow([sequence, plddt])
    