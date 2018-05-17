from module import Module
from variable import Variable
from torch import Tensor

class Linear(Module):

	def __init__(self, in_dim, out_dim, weight_init=None):
		super(Linear, self).__init__()
		self.in_dim = in_dim
		self.out_dim = out_dim
		self.bias = Variable(Tensor(out_dim, ))
		self.weight_init = weight_init
		
		if self.weight_init is None:
			self.weight = Variable(Tensor(out_dim, in_dim))
		elif self.weight_init == 'ones':
			ones_list = [1] * (in_dim*out_dim)
			self.weight = Variable(Tensor(ones_list).view(out_dim, in_dim))

	def forward(self , input):

		assert input.data.shape == (self.in_dim, )
		return Variable(self.weight.data @ input.data + self.bias.data)	
	
	def backward(self , gradwrtoutput):
		self.bias.grad = gradwrtoutput
		#self.weight.grad = gradwrtoutput @ 

	
	def param(self):
		return self.weight, self.bias