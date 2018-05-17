from module import Module
from torch import Tensor

class Linear(Module):

	def __init__(self, in_dim, out_dim, weight_init=None, bias_init='zero'):
		super(Linear, self).__init__()
		self.in_dim = in_dim
		self.out_dim = out_dim
		self.weight_init = weight_init
		self.bias_init = bias_init
		self.weight_grad = None
		self.bias_grad = None


	def init_weights(self, nb_samples):

		if self.weight_init is None:
			self.weight = Tensor(self.out_dim, self.in_dim)
		elif self.weight_init == 'ones':
			ones_list = [1] * (self.in_dim*self.out_dim)
			self.weight = Tensor(ones_list).view(self.out_dim, self.in_dim)
		self.weight_grad = Tensor(self.out_dim, self.in_dim)

		if self.bias_init is None:
			self.bias = Tensor(self.out_dim, nb_samples)
		elif self.bias_init == 'zero':
			zeros_list = [0] * (self.out_dim * nb_samples)
			self.bias = Tensor(zeros_list).view(self.out_dim, nb_samples)
		self.bias_grad = Tensor(self.out_dim, nb_samples)
		
	def forward(self , input):
		self.input = input
		#assert input.shape == (self.in_dim, )
		# input = (3, 2)
		# wt = (3, 3)
		# 
		# input = (in, 1) bias (out, 1)
		# input = (in, 2) bias (out, 2)
		return self.weight @ input + self.bias
	
	def backward(self , gradwrtoutput):
		# init them to zero in the loop as params.zero_grad()
		# Add then here I should accumulate
		self.bias_grad += gradwrtoutput
		nb_samples = gradwrtoutput.shape[1]
		self.weight_grad +=  gradwrtoutput.view(-1, nb_samples) @ self.input.view(nb_samples, -1)
		return self.weight.t() @ gradwrtoutput
	
	def param(self):
		return [self.weight, self.bias]

	def param_grad(self):
		return [self.weight_grad, self.bias_grad]

	def set_param_grad(self, param_grad):
		self.weight_grad = param_grad[0]
		self.bias_grad = param_grad[1]

	def update_param(self, lr):
		self.weight = self.weight - lr*self.weight_grad
		self.bias = self.bias - lr*self.bias_grad

