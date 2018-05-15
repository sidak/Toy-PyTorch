from module import Module
from variable import Variable

class Linear(Module):

	def __init__(self, in_dim, out_dim):
		super(Linear, self).__init__()
		self.in_dim = in_dim
		self.out_dim = out_dim
		self.weight = Variable((out_dim, in_dim))
		self.bias = Variable((out_dim, ))
		
	def forward(self , input):
		assert input.data.shape == (in_dim, )
		return self.weight @ input + self.bias	
	
	def backward(self , * gradwrtoutput):
		raise NotImplementedError
	
	def param(self):
		return self.weight, self.bias