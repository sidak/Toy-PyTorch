from module import Module
from torch import Tensor
import math


class Linear(Module):

	def __init__(self, in_dim, out_dim, weight_init='uniform_pos_neg', bias_init='zero'):
		""" in_di, and out_dim are the respective input and output dimensions sizes. The weight
			initialisation can be chosen via weight_init (see init_weights() for examples). The
			same holds for bias_init for the bias (see init_weights() for examples).

		"""
		super(Linear, self).__init__()
		self.in_dim = in_dim
		self.out_dim = out_dim
		self.weight_init = weight_init
		self.bias_init = bias_init
		self.weight_grad = None
		self.bias_grad = None
		self.weight = None
		self.bias = None

	def init_weights(self, nb_samples):
		""" Initialises the weights and the bias of the Layer. It requires the number of samples
			to be passes through the network, nb_samples.

			The following weight initialisations are available for both weight and bias:

			uniform_pos_neg: initialises the weights uniformly at random between [-1, 1]
			uniform_non_neg: initialises the weights uniformly at random between [0, 1]
			pytorch_default: initialises the weights uniformly at random between [-std, std]
							  where std = 1 / sqrt(self.in_dim)

			For the weight initialisation we additionally have:

			ones: initialises the weights to 1

			For the weight initialisation we additionally have:

			zero: initialises the weights to 0

		"""

		if self.weight_init is None:
			self.weight = Tensor(self.out_dim, self.in_dim)
		elif self.weight_init == 'ones':
			ones_list = [1] * (self.in_dim*self.out_dim)
			self.weight = Tensor(ones_list).view(self.out_dim, self.in_dim)
		elif self.weight_init == 'uniform_pos_neg':
			self.weight = Tensor(self.out_dim, self.in_dim).uniform_(-1, 1)
		elif self.weight_init == 'uniform_non_neg':
			self.weight = Tensor(self.out_dim, self.in_dim).uniform_(0, 1)
		elif self.weight_init == 'pytorch_default':
			stdv = 1. / math.sqrt(self.in_dim)
			self.weight = Tensor(self.out_dim, self.in_dim).uniform_(-stdv, stdv)

		self.weight_grad = Tensor(self.out_dim, self.in_dim)

		if self.bias_init is None:
			self.bias = Tensor(self.out_dim, 1)
		elif self.bias_init == 'zero':
			zeros_list = [0] * (self.out_dim * 1)
			self.bias = Tensor(zeros_list).view(self.out_dim, 1)
		elif self.bias_init == 'uniform_pos_neg':
			self.bias = Tensor(self.out_dim, 1).uniform_(-1, 1)
		elif self.bias_init == 'uniform_non_neg':
			self.bias = Tensor(self.out_dim, 1).uniform_(0, 1)
		elif self.bias_init == 'pytorch_default':
			stdv = 1. / math.sqrt(self.in_dim)
			self.bias = Tensor(self.out_dim, 1).uniform_(-stdv, stdv)

		self.bias_grad = Tensor(self.out_dim, 1)
		
	def forward(self , input):
		"""Return the output of the layer if ``input`` is input."""
		self.input = input
		return self.weight @ input + self.bias.repeat(1, input.shape[1])
	
	def backward(self , gradwrtoutput):

		self.bias_grad += gradwrtoutput.sum(dim=1,keepdim=True)
		nb_samples = gradwrtoutput.shape[1]
		
		for i in range(nb_samples):
			self.weight_grad += gradwrtoutput.narrow(1, i, 1) @ self.input.narrow(1, i, 1).t()

		return self.weight.t() @ gradwrtoutput
	
	def param(self):
		return [self.weight, self.bias]

	def param_grad(self):
		return [self.weight_grad, self.bias_grad]

	def set_param_grad(self, param_grad):
		self.weight_grad = param_grad[0]
		self.bias_grad = param_grad[1]

	def set_zero_grad(self):
		self.weight_grad.zero_()
		self.bias_grad.zero_()

	def update_param(self, lr, wd=0):
		""" Performs the gradient step once the forwards and backward passes have been done. """
		self.weight = self.weight * (1 - lr*wd) - lr*self.weight_grad
		self.bias = self.bias - lr*self.bias_grad