import numpy as np, os 
from scipy.spatial.distance import pdist, squareform

__all__ = ['extract_data']

def extract_data(a3mfname, cffname):
    """
    INPUTS:
        a3mfname <str> - The a3m file.
        cffname <str>  - The cf file.
        
    RETURNS:
        1. msa <numpy.ndarray> - 
        2. W <numpy.ndarray> -
        3. nat_contact <>
    """
    assert os.path.exists(a3mfname) and os.path.exists(cffname)
    names, seqs = parse_a3m(a3mfname, a3m=True)  # parse sequences
    msa = create_msa(seqs)                # convert sequences to one-hot
    W = calculate_seq_weights(msa)        # weight per sequence 
    nat_contacts = parse_cf(cffname)  # get contacts from .cf file
    return msa, W, nat_contacts

def parse_a3m(filename, a3m=False):
    '''function to parse fasta file'''

    if a3m:
        import string
        # for a3m files the lowercase letters are removed
        # as these do not align to the query sequence
        rm_lc = str.maketrans(dict.fromkeys(string.ascii_lowercase))
    
    header, sequence = [],[]
    lines = open(filename, "r")
    for line in lines:
        line = line.rstrip()
        if line[0] == ">":
            header.append(line[1:])
            sequence.append([])
        else:
            if a3m: line = line.translate(rm_lc)
            else: line = line.upper()
            sequence[-1].append(line)
    lines.close()
    sequence = [''.join(seq) for seq in sequence]

    return header, sequence


def create_msa(seqs, alphabet="ARNDCQEGHILKMFPSTWYV-"):
    '''one hot encode msa'''
    states = len(alphabet)  

    # create dictionary of alphabet
    a2n = {a:n for n, a in enumerate(alphabet)}

    # get indices from dictionary for each AA at each position
    msa_ori = np.array([[a2n.get(aa, states-1) for aa in seq] for seq in seqs])

    # return one-hot
    return np.eye(states)[msa_ori]


def calculate_seq_weights(msa, eff_cutoff=0.8):
    '''compute weight per sequence'''
    if msa.ndim == 3: msa = msa.argmax(-1)    
    msa_sm = 1.0 - squareform(pdist(msa, "hamming"))
    weights = 1/(msa_sm >= eff_cutoff).astype(np.float).sum(-1)
    return weights

def parse_cf(filename, cutoff=0.001):
    # get contacts
    # contact Y,1     Y,2     0.006281        MET     ARG

    # parse contact file
    n, cons = 0, []
    for line in open(filename, "r"):
        line = line.rstrip()
        if line[:7] == "contact":
            _, _, i, _, j, p, _, _ = line.replace(",", " ").split()
            i, j, p = int(i), int(j), float(p)
            if i > n: 
                n = i
            if j > n: 
                n = j
            cons.append([i-1, j-1, p])

    # create contact map
    cm = np.zeros([n, n])
    for i, j, p in cons: 
        cm[i,j] = p
    return (cm + cm.T)