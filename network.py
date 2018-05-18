from module import Module
from torch import Tensor

class Network(Module):

	def __init__(self, layers, nb_samples):
		super(Network, self).__init__()
		self.layers = layers
		self.nb_samples = nb_samples
		self.init_weights()
		
	def forward(self , input):
		inp = input
		for layer in self.layers:
			out = layer.forward(inp)
			inp = out
		return out

	def backward(self , gradwrtoutput):
		grad = gradwrtoutput
		for layer in reversed(self.layers):
			grad = layer.backward(grad)

	def _get_components(self, tens_shape):
		components = 1
		for dm in tens_shape:
			components *= dm
		return components

	def _set_zero(self, tens):
		tens_shape = tens.shape
		zeros_list = [0.0] * self._get_components(tens_shape)
		tens = Tensor(zeros_list).view(tens_shape)
		return tens

	def init_weights(self):
		for layer in self.layers:
			layer.init_weights(self.nb_samples)

	def zero_grad(self):
		for layer in self.layers:
			#layer.set_param_grad([self._set_zero(par_grad) for par_grad in layer.param_grad()])
			layer.set_zero_grad()
			
	def grad_step(self, lr):
		for layer in self.layers:
			layer.update_param(lr)

	