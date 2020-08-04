import numpy as np 

import torch
from torch import nn

# entire training process
def train(model, 
		  optimizer, 
		  lossfn, 
		  xdata, ydata, 
		  regularizer=None, 
		  batch_size=32, 
		  maxiter=15000,
		  check_every=500):
	"""
	INPUTS:
		1. model <torch.tensor> - 
		2.
		3.
		4.
		5.
		6.
		7.
	"""

# one step of the training 
def trainstep(model, 
              lossfn,  
              optimizer, 
              xbatch, 
              ybatch, 
              weights=None,
              regularizer=None):
    """
    Perform one step of model training. 
    
    INPUTS:
        1. model <torch.nn.Module> - The module with trainable parameters. 
        2. lossfn <callable> - The objective function to be minimized ; computes the loss per example. 
        3. optimizer <torch.optim.Optimizer> - An instance of a torch optimizer.
        4. xbatch <torch.tensor> - A batch of input data.
        5. ybatch <torch.tensor> - Target labels. 
        6. weights <array> - Weights associated with each sample in the mini-batch.  
        7. regularizer <callable> - Takes the 
    """
    # get the batch size and weight per example 
    batch_size = xbatch.shape[0]
    if weights == None:
        weights = torch.ones(batch_size)
    
    # zero out the gradients
    optimizer.zero_grad()
    
    # get the predictions 
    ypred = model(xbatch)   
    
    # get the per sample loss
    per_sample_losses = lossfn(ypred, ybatch)  # should be a row vector 
    
    # get the weighted loss 
    weighted_loss = per_sample_losses * weights
    
    # get the mean loss
    loss = weighted_loss.mean()
    
    # get the regularized loss
    if regularizer == None:
        reg_loss = loss
    else:
        reg_loss = loss + regularizer(model)
    
    # compute gradients 
    reg_loss.backward()
    
    # update the model parameters 
    opt.step()
    
    return loss, reg_loss

