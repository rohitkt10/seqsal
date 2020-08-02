import numpy as np, os 
import torch
from torch import nn
from sklearn.cross_validation import train_test_split

__all__ = ['ProteinMSADataset']

class ProteinMSADataset(torch.utils.data.Dataset):
	"""
	A PyTorch dataset class for protein sequence models. 

	The dataset must be an .npz file with the following keys:
	1. X - a N x L x A tensor of MSA data.
	2. W - a row vector of length N, containing the weights of each sequence.

	Querying the dataset returns one hot representation of the requested 
	sequences and their associated weights.  
	"""
	def __init__(self, datapath):
		super(ProteinMSADataset, self).__init__()
		assert os.path.isfile(datapath), 'No file found at given location.'
		self.datapath = datapath
		datafile = np.load(self.datapath)
		self.X = datafile['X']
		Xonehot = np.eye(21)[self.X]
		shape = (Xonehot.shape[0], Xonehot.shape[2], Xonehot.shape[1])
		self.Xonehot = Xonehot.reshape(*shape)
		self.W = datafile['W']
		self.contacts = datafile['cons']
	
	def __len__(self):
		return len(self.Xonehot)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		xs = self.Xonehot[idx]
		ws = self.W[idx]
		return xs, ws
