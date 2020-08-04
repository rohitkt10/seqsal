import numpy as np, os
import scipy
from scipy.spatial.distance import pdist, squareform


def get_eff(msa, eff_cutoff=0.8):
    """
    Compute weight per sequence

    INPUTS:
        1. msa <numpy.ndarray> - Multiple sequence alignment.
        2. eff_cutoff <float> -
    """
    if msa.ndim == 3: msa = msa.argmax(-1)
    msa_sm = 1.0 - squareform(pdist(msa,"hamming"))
    msa_weights = 1/(msa_sm >= eff_cutoff).astype(np.float).sum(-1)
    return msa_weights

def pw_saliency(model):
    """
    given pytorch model, compute pariwise term.
    """
    out = model.output[:,i]
    L,A = [int(s) for s in model.output.get_shape()[1:]]
    sal = -K1.gradients(-K.sum(np.eye(A)*K.log(out + 1e-8)), model.input)[0]
    null = np.zeros((A,L,A))
    pw = np.array([sess.run(sal, {i:j, model.input:null}) for j in range(L)])
    return 0.5*(pw+np.transpose(pw,(2,3,0,1)))


if __name__=='__main__':
    data = np.load('data.npz')
    msa = data['X']


