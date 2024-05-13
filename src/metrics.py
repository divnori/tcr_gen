from collections import Counter, OrderedDict
import pandas as pd
import random
import torch
import torch.nn as nn
import subprocess
from Bio import pairwise2
from Bio.Align import substitution_matrices
import itertools
from tqdm import tqdm

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

def get_gen_seqs(gen_file):
    with open('data/trbc2_seq.txt', 'r') as file:
        const_region = file.read().strip()

    df = pd.read_csv(gen_file, header=None)
    full_seqs = df[0].tolist()
    cdr_lens = df[1].astype(int).tolist()
    const_idxs = [x.find(const_region) for x in full_seqs]
    gen_seqs = [full_seqs[i][const_idxs[i] - (cdr_lens[i]+4):const_idxs[i]] for i in range(len(full_seqs))]
    df[len(df.columns)] = gen_seqs
    df.to_csv(gen_file, header=None)
    return gen_seqs

def get_true_seqs(true_file='data/TRB_CDR3_human_VDJdb.tsv'):
    true_df = pd.read_csv(true_file, delimiter='\t', error_bad_lines=False)
    true_seqs = true_df['CDR3'].tolist()

    return true_seqs

def calc_unconditiona_kl(gen_seqs):

    true_seqs = get_true_seqs()

    print(f"Number of Sequences in Reference Distribution: {len(true_seqs)}")
    print(f"Number of Sequences in Generated Distribution: {len(gen_seqs)}")
    
    kl_loss = nn.KLDivLoss(reduction="sum")
    kl_value = kl(true_seqs, gen_seqs, kl_loss)
    print(f"KL Divergence = {kl_value}")

def calc_olga(gen_file, lstm=False, evodiff=False, native=True):
    chain = 'humanTRB' if 'trb' in gen_file else 'humanTRA'

    if lstm:
        df = pd.read_csv(gen_file, header=None)
        seqs = df[0].tolist()
    elif evodiff:
        df = pd.read_csv(gen_file)
        seqs = df["sequence"].tolist()
        left_context = df["left_context"].tolist()
        gen_lens = df["generated_len"].tolist()
        seqs = [s[len(lc):len(lc)+gl] for s, lc, gl in zip(seqs, left_context, gen_lens)]
    elif native:
        df = pd.read_csv(gen_file)
        seqs = df["CDR3"].tolist()
    
    probs = 0
    for seq in tqdm(seqs):
        command = [
            f"olga-compute_pgen --humanTRB {seq}"
        ]
        result = subprocess.run(command, cwd="OLGA/olga", shell=True, capture_output=True, text=True)

        prob = float(result.stdout.split(" ")[7].split('\n')[0])
        probs += prob

    print(f"Average Probability: {probs/len(seqs)}")


def calc_seq_diversity(gen_seqs):
    # is average fine or distribution better?
    matrix = substitution_matrices.load("BLOSUM62")
    total_score = 0
    num_pairs = 0

    for s1, s2 in tqdm(itertools.combinations(gen_seqs, 2)):
        length = max(len(s1), len(s2))
        for a in pairwise2.align.globaldx(s1, s2, matrix):
            total_score += a.score/length
            num_pairs += 1

    print(f"Pairwise Similarity = {total_score/num_pairs}")

if __name__ == "__main__":

    gen_file = 'results/trbv_gen.csv'
    
    # # lstm
    # df = pd.read_csv(gen_file, header=None)
    # gen_seqs = df[0].tolist()
    # gen_seqs = random.sample(gen_seqs, 100)

    # evodiff
    df = pd.read_csv(gen_file)
    seqs = df["sequence"].tolist()
    left_context = df["left_context"].tolist()
    gen_lens = df["generated_len"].tolist()
    gen_seqs = [s[len(lc):len(lc)+gl] for s, lc, gl in zip(seqs, left_context, gen_lens)]
    gen_seqs = random.sample(gen_seqs, 100)

    # calc_olga(gen_file)
    calc_seq_diversity(gen_seqs)
    
