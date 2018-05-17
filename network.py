from module import Module
from variable import Variable
from torch import Tensor

class Network(Module):

	def __init__(self, layers):
		super(Network, self).__init__()
		self.layers = layers
		#self.
		#self.out_dim = out_dim
		#self.weight = Variable(Tensor(out_dim, in_dim))
		#self.bias = Variable(Tensor(out_dim, ))
		
	def forward(self , input):
		inp = input
		for layer in self.layers:
			out = layer.forward(inp)
			inp = out
		return out

	def backward(self , gradwrtoutput):
		#self.bias.grad = gradwrtoutput
		#self.weight.grad = gradwrtoutput @ 
		pass
