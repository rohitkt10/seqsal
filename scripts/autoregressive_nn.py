import numpy as np 
import torch
from torch import nn


__all__ = ['MaskedAutoEncoder']

class MaskedLinear(nn.Linear):
	"""
	An implementation of a fully connected NN layer coupled with a 
	binary mask to enforce autoregressive structure in the NN output. 
	"""
	def __init__(self, in_features, out_features, mask, bias=True):
		super(MaskedLinear, self).__init__(in_features, out_features, bias)
		if not isinstance(mask, torch.Tensor):
			mask = torch.tensor(mask, dtype=torch.float32)
		self.register_buffer(name='mask', tensor=mask.data)
	def forward(self, x):
		masked_weight = self.weight * self.mask
		return nn.functional.linear(x, masked_weight, self.bias)


class MaskedAutoEncoder(nn.Module):
	"""
	An implementation of the autoregressive NN as described by Germain et al (2015). 

	See:
	   MADE: Masked Autoencoder for Distribution Estimation.
	   https://arxiv.org/pdf/1502.03509.pdf
	"""
	def __init__(self, D, Hs, 
		               num_outputs=None, 
		               permutation=None, 
		               actfn=nn.ReLU(),
		               bias=True):
		"""
		INPUTS:
			1. D <int> - The dimensionality of the input.
			2. Hs <list of int> - A list containing the sizes of the hidden layers.
			3. num_outputs <int> - The number of outputs that need to have the autoregressive
									structure. For example, when modeling affine Gaussian flows, 
									the network needs to output mean and log std. with autoregressive 
									structure.
			4. permutation <array of int> - The ordering of the conditional structure.
											By default, the identity permutation is used 
											which leads to triangular Jacobian matrix.
			5. actfn <callable> - Activation function (not applied to last layer)
			6. bias <bool>
		"""
		super(MaskedAutoEncoder, self).__init__()
		if permutation is None:
			permutation = np.arange(D)
		if num_outputs is None:
			num_outputs = 1
		
		self.actfn = actfn
		
		self.masks = get_all_masks(D, Hs, permutation)
		hs = [D] + Hs + [D]
		layers = []
		for i, (h_in, h_out) in enumerate(zip(hs[:-1], hs[1:])):
			if i == len(self.masks) - 1:
				mask = torch.cat([self.masks[i]]*num_outputs, 0)
				layer = MaskedLinear(h_in, num_outputs*h_out, mask)
			else:
				mask = self.masks[i]
				layer = MaskedLinear(h_in, h_out, mask)
			layers.append(layer)
		self.layers = nn.ModuleList(layers)

	def forward(self, x):
		y = x
		for i, layer in enumerate(self.layers):
			y = layer(y)
			if i < len(self.layers) - 1:
				y = self.actfn(y)
		return y


def get_mask(mprev, m, final=False):
    D, K = len(mprev), len(m)
    Mprev = np.tile(mprev, (K, 1))
    M = np.tile(m, (D, 1)).T
    if final:
        return (M > Mprev).astype(np.float32)
    else:
        return (M >= Mprev).astype(np.float32)

def get_all_masks(D, K, permutation):
    """
    Alg. 1 of Germain et al (2015).
    
    INPUTS:
        1. D <int> - The size of the input/output. 
        2. K <list of ints> - The number of hidden units per hidden layer.  
    """
    L = len(K)
    # sample the input layer mask vector 
    #m0 = np.random.permutation(np.arange(1, D+1))
    m0 = np.arange(1, D+1)[permutation]  
    ms = [m0]
    
    # get all the remaining mask vectors
    for l in range(L):
        mprev = ms[-1]
        mprev_min = mprev.min()
        ml = np.random.choice(np.arange(mprev_min, D), size=((K[l],)))
        ms.append(ml)
    
    # construct all the masks 
    masks = []
    for l in range(1, L+1):
        mask = get_mask(ms[l-1], ms[l])
        masks.append(torch.tensor(mask, dtype=torch.float32))
    final_mask = get_mask(ms[-1], m0, final=True)
    masks.append(torch.tensor(final_mask, dtype=torch.float32))
    
    return masks


def test_made_implementation():
	"""
	Tests the implementation of the masked autoencoder.

	Tests:
	1. Test that the matrix product of all the masks in reverse 
	order, i.e., M_L \\cdot M_{L-1} \\cdot ... \\cdot M_1 
	is a lower triangular matrix. 
	"""

	# TEST 1 : Test for lower triangularity of the mask. 
	D = 5
	made = MaskedAutoEncoder(D, Hs=[30, 30], num_outputs=1)
	masks = [m.data.numpy() for m in made.masks]
	mask_prod = np.linalg.multi_dot(masks[::-1])
	test1_status = np.allclose(mask_prod, np.tril(mask_prod))

	# TEST 2 : Jacobian case 1 : num_outputs = 1
	x = torch.rand(1, D, requires_grad=True)
	y = made(x)
	dydxs = []
	for i in range(D):                                                                                                                     
	    output = torch.zeros(1, D)                                                                                                          
	    output[:, i] = 1.                                                                                                                     
	    dydxs.append(grad(y1, x, grad_outputs=output, retain_graph=True)[0])
	J=torch.cat(dydxs).data.numpy()
	test2_status = np.allclose(J, np.tril(J))

	# TEST 3 : Jacobian case 2 : num_output > 1 
	made = MaskedAutoEncoder(D, Hs=[30, 30], num_outputs=2)
	x = torch.rand(1, D, requires_grad=True)
	y = made(x)
	y1, y2 = y[:, :D], y[:, D:]
	dy1dxs = []
	for i in range(D):                                                                                                                     
	    output = torch.zeros(1, D)                                                                                                          
	    output[:, i] = 1.                                                                                                                     
	    dy1dxs.append(grad(y1, x, grad_outputs=output, retain_graph=True)[0])
	J1=torch.cat(dy1dxs).data.numpy()

	dy2dxs = []
	for i in range(D):                                                                                                                     
	    output = torch.zeros(1, D)                                                                                                          
	    output[:, i] = 1.                                                                                                                     
	    dy2dxs.append(grad(y2, x, grad_outputs=output, retain_graph=True)[0])
	J2=torch.cat(dy2dxs).data.numpy()
	test3_status = np.allclose(J1, np.tril(J1)) and np.allclose(J2, np.tril(J2))

	tests_status = test1_status and test2_status and test3_status

	if tests_status:
		print('MADE implementation passed tests.')
	else:
		print('MADE implementation did not pass tests')