from collections import Counter, OrderedDict
import pandas as pd
import torch
import torch.nn as nn

def seqs_to_dict(seqs):
    seqs = ''.join(seqs)
    aminos_gen = Counter(
        {'A': 0, 'M': 0, 'R': 0, 'T': 0, 'D': 0, 'Y': 0, 'P': 0, 'F': 0, 'L': 0, 'E': 0, 'W': 0, 'I': 0, 'N': 0, 'S': 0, \
         'K': 0, 'Q': 0, 'H': 0, 'V': 0, 'G': 0, 'C': 0, 'X': 0, 'B': 0, 'Z': 0, 'U': 0, 'O': 0, 'J': 0, '-': 0})
    aminos_gen.update(seqs)

    order_of_keys = ['A','M','R','T','D','Y','P','F','L','E','W','I','N','S',
                     'K','Q','H','V','G','C','X','B','Z','J','O','U','-']
    list_of_tuples = [(key, aminos_gen[key]) for key in order_of_keys]
    aminos_gen_ordered = OrderedDict(list_of_tuples)
    return aminos_gen_ordered

def normalize_list(list):
    norm = sum(list)
    new_list = [item / norm for item in list]
    return new_list

def kl(ref_seqs, gen_seqs, kl_loss):
    true_aas = seqs_to_dict(ref_seqs)
    gen_aas = seqs_to_dict(gen_seqs)

    a_list = list(true_aas.values())
    a = normalize_list(a_list)

    b_list = list(gen_aas.values())
    b = normalize_list(b_list)

    kl = kl_loss(torch.tensor(a[0:21]).log(), torch.tensor(b[0:21])).item()
    return kl

def calc_unconditiona_kl(gen_file):

    with open('data/trbc2_seq.txt', 'r') as file:
        const_region = file.read().strip()

    df = pd.read_csv(gen_file, header=None)
    full_seqs = df[0].tolist()
    cdr_lens = df[1].astype(int).tolist()
    const_idxs = [x.find(const_region) for x in full_seqs]
    gen_seqs = [full_seqs[i][const_idxs[i] - (cdr_lens[i]+4):const_idxs[i]] for i in range(len(full_seqs))]

    true_df = pd.read_csv('data/TRB_CDR3_human_VDJdb.tsv', delimiter='\t', error_bad_lines=False)
    true_seqs = true_df['CDR3'].tolist()

    print(f"Number of Sequences in Reference Distribution: {len(true_seqs)}")
    print(f"Number of Sequences in Generated Distribution: {len(gen_seqs)}")
    
    kl_loss = nn.KLDivLoss(reduction="sum")
    kl_value = kl(true_seqs, gen_seqs, kl_loss)
    print(f"KL Divergence = {kl_value}")

if __name__ == "__main__":

    calc_unconditiona_kl('results/trbv_gen.csv')

    

