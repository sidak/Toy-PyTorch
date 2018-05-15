from module import Module
from variable import Variable
from torch import Tensor
class Linear(Module):

	def __init__(self, in_dim, out_dim):
		super(Linear, self).__init__()
		self.in_dim = in_dim
		self.out_dim = out_dim
		self.weight = Variable(Tensor((out_dim, in_dim)))
		self.bias = Variable(Tensor((out_dim, )))
		
	def forward(self , input):
		assert input.data.shape == (self.in_dim, )
		return self.weight.data @ input.data + self.bias	
	
	def backward(self , * gradwrtoutput):
		raise NotImplementedError
	
	def param(self):
		return self.weight, self.bias