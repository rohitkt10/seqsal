import torch
from torch import nn 

__all__ = ['NeuralNetwork', 'LinearAutoEncoder', 'VariationalAutoEncoder', 'LpRegularizer', 'Reshape']


class NeuralNetwork(nn.Module):
	def __init__(self, Hs, actfn=nn.ReLU(), lastlayeractfn=nn.Identity()):
		"""
		Hs <list/tuple>: The size of the input, hidden layers and output. For example, 
						 for a NN with input size 3, output size 2, and 2 hidden layers 
						 with 20 units, Hs = [3, 20, 20, 2]. 
		actfn <callable>: The activation function to use in the intermediate layers.
		lastlayerfn <callable>: The activation function to use in the final layer
		"""
		super(NeuralNetwork, self).__init__()
		assert len(Hs) > 2, 'Atleast 1 hidden layer required.'
		self.Hs = Hs
		self.actfn = actfn
		self.lastlayeractfn = lastlayeractfn

		h_pairs = zip(self.Hs[:-1], self.Hs[1:])
		layers = []
		for i, pair in enumerate(h_pairs):
			h_in, h_out = pair 
			layer = nn.Linear(h_in, h_out)
			layers.append(layer)
		self.layers = nn.ModuleList(layers)

	def forward(self, x):
		y = x
		num_layers = len(self.layers)
		for i, layer in enumerate(self.layers):
			y = layer(x)
			if i < num_layers - 1:
				y = self.actfn(x)
			else:
				y = self.lastlayeractfn(x)
		return y

class LpRegularizer(nn.modules.loss._Loss):
	def __init__(self, p=2, dtype=torch.float32):
		"""
		Compute the p-norm regularizer for all 
		parameters in a nn.Module object.  

		||W||_p = (\\sum_i \\abs W_{i}^{p})^{1/p}

		INPUTS:
			p <int> - The norm order. 
		
		"""
		self.p=p
		self.dtype = dtype

	def forward(module):
		loss = torch.tensor(0., dtype=self.dtype)
		for m in module.modules():
			if hasattr(m, 'weight'):
				loss += m.weight.norm(p=self.p)
		return loss

class Reshape(nn.Module):
	def __init__(self, target_shape):
		"""
		The target shape to reshape incoming tensors to. 

		Ex.
		>> x = torch.randn(10, 3, 2)
		>> layer = Reshape((6,))
		>> y = Reshape(x)     # shape == (10, 6)

		"""
		super(Reshape, self).__init__()
		self.target_shape = list(target_shape)

	def forward(self, x):
		xshape = list(x.shape)
		yshape = [xshape[0]] + self.target_shape
		y = x.reshape(*yshape)
		return y

class LinearAutoEncoder(nn.Module):
	"""
	Implementation of a linear autoencoder,i.e., a NN of the form:

	f(x) = W_2^T z + b_2, x \\in \\R^{D} 
	where, z = W_1^T + b_1 \\in \\R^{d}. 

	"""
	def __init__(self, L, A, rank=256, bias=False):
		"""
		INPUTS:
		    1. L <int> - Sequence length.
		    2. A <int> - Vocabulary size.
		    3. rank <int> - Size of latent space encoding.
			5. use_e <bool> 
			6. bias <bool>
		"""
		super(LinearAutoEncoder, self).__init__()
		self.L = L
		self.A = A

		# define the layers 
		self.flatten = nn.Flatten()
		self.encoder = nn.Linear(in_features=L*A, out_features=rank, bias=bias)
		self.decoder = nn.Linear(in_features=rank, out_features=L*A)
		self.reshape = Reshape(target_shape=[A, L])
		self.softmax = nn.Softmax(dim=1)

	def forward(self, x):
		y = x
		y = self.flatten(y)
		y = self.encoder(y)
		y = self.decoder(y)
		y = self.reshape(y)
		y = self.softmax(y)
		return y

class VariationalAutoEncoder(nn.Module):
    def __init__(self, L, A, 
                 encoder_units=[],
                 encoder_actfn=nn.ReLU(),
                 decoder_units=[],
                 decoder_actfn=nn.ReLU(),
                 rank=256,
                 drop=0.5):        
        """
        A variational autoencoder with mean field Gaussian 
        approximation of the posterior over the latent variables. 

        INPUTS:
            1. L   <int> - Length of sequence.
            2. A   <int> - Size of vocabulary.
            3. encoder_units <tuple/list> - Sizes of hidden layers in the encoder network.
            4. encoder_actfn <torch.nn.Module> - The activation function to be used in 
                                                 the encoder network.
            5. decoder_units <tuple/list> - Sizes of hidden lauers in the decoder network.
            6. decoder_actfn <torch.nn.Module> - The activation function to be used in 
                                                  the decoder network. 
            7. rank <int> - Size of the encoding.
            8. drop <float> The dropout probability. 
        """
        super(VariationalAutoEncoder, self).__init__()
        self.rank = rank
        F = L*A

        ## define the layers 

        # flatten layer 
        self.flatten = nn.Flatten()

        # encoder layers
        encoder_layers = []
        unit_sizes = [F]
        for i, units in enumerate(encoder_units):
            fclayer = nn.Linear(unit_sizes[-1], units)
            unit_sizes.append(units)
            encoder_layers.append(fclayer)
            encoder_layers.append(encoder_actfn)
            encoder_layers.append(nn.Dropout(p=drop))
            encoder_layers.append(nn.BatchNorm1d(num_features=units))
        z_dist_params_layer = nn.Linear(unit_sizes[-1], 2*rank)
        encoder_layers.append(z_dist_params_layer)
        self.encoder = nn.Sequential(*encoder_layers)
        
        # decoder (i.e. generative network)
        # the input is samples of the latent variable (batch_shape x rank)
        # the output is of shape (batch_shape x # AA x seq length)
        decoder_layers = []
        unit_sizes = [rank]
        for i, units in enumerate(decoder_units):
            fclayer = nn.Linear(unit_sizes[-1], units)
            unit_sizes.append(units)
            decoder_layers.append(fclayer)
            decoder_layers.append(decoder_actfn)
            decoder_layers.append(nn.Dropout(p=drop))
            decoder_layers.append(nn.BatchNorm1d(num_features=units))
        decoding_layer = nn.Linear(unit_sizes[-1], F)
        decoder_layers.append(decoding_layer)
        decoder_layers.append(Reshape(target_shape=[L, A]))
        decoder_layers.append(nn.Softmax(dim=2))
        self.decoder = nn.Sequential(*decoder_layers)
        
        # prior distribution over z 
        self.logpriorsigma = nn.Parameter(data=torch.ones((self.rank,)))
        priorsigma = torch.exp(self.logpriorsigma)
        self.zprior = dist.Independent(
            dist.Normal(
                loc=torch.zeros((self.rank,)), 
                scale=priorsigma * torch.ones((self.rank,))
            ),
            reinterpreted_batch_ndims=1
        )
        
    def forward(self, x):
        """
        Given a min-batch of data-samples, return the:
        	1. sample drawn from the approximate latent posterior (torch.tensor).
            2. the approximate posterior (torch.distributions.Distribution).
            3. conditional distribution of the reconstruction conditional 
               on the sampled latent variable (torch.distributions.Distribution).
        
        For simplicity we will decode with 1 sample from the latent posterior following the same 
        procedure as Kingma & Welling (2014). 
        
        Reference:
          Auto-Encoding Variational Bayes, Kingma & Welling, 2014. 
          https://arxiv.org/pdf/1312.6114.pdf
        """
        # encoding 
        z_params = self.encoder(self.flatten(x)) 
        z_mu, z_log_scale = z_params[:, :self.rank], z_params[:, self.rank:]
        z_scale = torch.exp(z_log_scale)
        
        # latent posterior 
        zdist = dist.Independent(dist.Normal(loc=z_mu, scale=z_scale), reinterpreted_batch_ndims=1)
        
        ####
        ### The normalizing flow part goes here ###
        ### zdist is a mean field Gaussian posterior 
        ### Transform zdist by passing it through bijectors
        ####

        # sample from the latent posterior
        zsample = zdist.sample()
        
        # get the categorical probabilities for the output 
        xprobs = self.decoder( zsample )
        
        # likelihood of the output conditional on the sampled LV 
        x_cond_z_dist = dist.Independent(dist.Categorical(probs=xprobs), reinterpreted_batch_ndims=1)
        
        return zsample, zdist, x_cond_z_dist
    
    def ELBO(self, xbatch, scale_factor=1.):
        """
        INPUTS:
            1. xbatch <torch.tensor> - A batch of training data. 
            2. scale_factor <float> - A scaling factor to account for the use of mini-batches. 
        """
        if not isinstance(scale_factor, torch.Tensor):
            scale_factor = torch.tensor(scale_factor, dtype=DTYPE)
        
        zsamples, zdist, x_cond_z_dist = self.forward(xbatch) 
        
        # entropy term 
        entropy = zdist.entropy()  # should be an array of size (N,)  
        
        # reconstruction probability - given the sample from the approximate posterior,
        # get the joint probability of the model 
        logprior = self.zprior.log_prob(zsamples)  # should be of size (N,)
        loglike  = x_cond_z_dist.log_prob(xbatch.argmax(dim=2))   # should be of size (N,)
        logjoint = loglike + logprior   # should be of size (N,)
        
        # elbo 
        elbo = scale_factor * (entropy + logjoint).sum()
        
        return elbo
    
    def negELBO(self, xbatch, scale_factor=1.):
        return -self.ELBO(xbatch, scale_factor)
    
    def sample(self, num_samples=1):
        """
        Sample from the generative model.
        
        The generative process is:
        
        1. z \\sim p(z) = N(0, \\sigma^2 * I)
        2. x | z \\sim p(x|z) = Cat( x | decoder(z) )
        """
        # make sure to take the model to eval mode 
        if self.training:
            self.eval()
        
        if num_samples == 1:
            zsample = self.zprior.sample()
            xprobs = self.decoder( zsample[None, :] )
            xdist  = dist.Categorical(probs=xprobs)
            return xdist.sample()[0]
        else:
            zsamples = self.zprior.sample((num_samples,))
            xprobs = self.decoder(zsamples)
            xdist = dist.Independent(dist.Categorical(probs=xprobs), reinterpreted_batch_ndims=1)
            return xdist.sample()


if __name__ == '__main__':
	import numpy as np
	from utils import get_eff
	data = np.load('data.npz')
	X = np.eye(21)[data['X']]
	W = data['W']
	lae = LinearAutoEncoder()
