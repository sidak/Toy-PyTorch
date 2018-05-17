from module import Module
from torch import Tensor

class Tanh(Module):

	def __init__(self):
		super(Tanh, self).__init__()
		self.output = None

	def forward(self, input):
		self.output = self._tanh(input)
		return self.output

	def _tanh(self, x):
		return 1.0 - (2.0/ (1.0 + (2.0*x).exp()))

	def backward(self, gradwrtoutput):
		#self.bias.grad = gradwrtoutput
		#self.weight.grad = gradwrtoutput @ 
		return gradwrtoutput * (1.0 - (self._tanh(self.output) * self._tanh(self.output)))
	

class Relu(Module):

	def __init__(self):
		super(Relu, self).__init__()
		self.output = None

	def forward(self , input):
		self.output = self._relu(input)
		return self.output

	def _relu(self, x):
		return (x.abs() + x)/2.0

	def backward(self , gradwrtoutput):
		#self.bias.grad = gradwrtoutput
		#self.weight.grad = gradwrtoutput @ 
		return gradwrtoutput * self._relu(self.output)