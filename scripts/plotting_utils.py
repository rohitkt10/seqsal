import numpy as np 
import matplotlib
import matplotlib.pyplot as plt

__all__ = [
			'visualize_mask'
		  ]

def visualize_mask(M):
    """
    M -> Binary mask
    """
    nrows, ncols = M.shape
    plt.figure(figsize=(7, 7))
    plt.spy(M)
    plt.xticks(np.arange(ncols)-0.5, np.arange(1, ncols+1), fontsize=15)
    plt.yticks(np.arange(nrows)-0.5, np.arange(1, nrows+1), fontsize=15)
    plt.grid(color='white', linestyle="-", linewidth=2)