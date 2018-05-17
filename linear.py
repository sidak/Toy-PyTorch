from module import Module
from variable import Variable
from torch import Tensor

class Linear(Module):

	def __init__(self, in_dim, out_dim, weight_init=None, bias_init='zero'):
		super(Linear, self).__init__()
		self.in_dim = in_dim
		self.out_dim = out_dim
		self.weight_init = weight_init
		self.bias_init = bias_init

		if self.weight_init is None:
			self.weight = Variable(Tensor(out_dim, in_dim))
		elif self.weight_init == 'ones':
			ones_list = [1] * (in_dim*out_dim)
			self.weight = Variable(Tensor(ones_list).view(out_dim, in_dim))

		if self.bias_init is None:
			self.bias = Variable(Tensor(out_dim, ))
		elif self.bias_init == 'zero':
			zeros_list = [0] * (out_dim)
			self.bias = Variable(Tensor(zeros_list).view(out_dim, ))

	def forward(self , input):
		self.input = input
		assert input.data.shape == (self.in_dim, )
		return Variable(self.weight.data @ input.data + self.bias.data)	
	
	def backward(self , gradwrtoutput):
		# init them to zero in the loop as params.zero_grad()
		# Add then here I should accumulate
		self.bias.grad = gradwrtoutput
		self.weight.grad =  self.input.data @ gradwrtoutput.t()
		return self.weight.data.t() @ gradwrtoutput
	
	def param(self):
		return self.weight, self.bias