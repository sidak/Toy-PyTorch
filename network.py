from module import Module
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
			#print("out is ", out)
			inp = out
		return out

	def backward(self , gradwrtoutput):
		#self.bias.grad = gradwrtoutput
		#self.weight.grad = gradwrtoutput @ 
		grad = gradwrtoutput
		for layer in reversed(self.layers):
			grad = layer.backward(grad)

	def _get_components(tens_shape):
		components = 1
		for dm in tens_shape:
			components *= dm
		return dm

	def _set_zero(tens):
		tens_shape = tens.shape
		zeros_list = [0] * _get_components(tens_shape)
		tens = Tensor(zeros_list).view(tens_shape)
		return tens

	def zero_grad(self):
		for layer in layers:
			for par_grad in layer.param_grad():
				par_grad = _set_zero(par_grad)
