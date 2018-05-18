from module import Module
from torch import Tensor

class Tanh(Module):

	def __init__(self):
		super(Tanh, self).__init__()
		self.input = None

	def forward(self, input):
		self.input = input
		return self._tanh(input)

	def _tanh(self, x):
		return 1.0 - (2.0/ (1.0 + (2.0*x).exp()))

	def backward(self, gradwrtinput):
		#self.bias.grad = gradwrtinput
		#self.weight.grad = gradwrtinput @ 
		return gradwrtinput * (1.0 - (self._tanh(self.input) * self._tanh(self.input)))


class Relu(Module):

	def __init__(self):
		super(Relu, self).__init__()
		self.input = None

	def forward(self , input):
		self.input = input
		return self._relu(input)

	def _relu(self, x):
		return (x.abs() + x)/2.0

	def backward(self , gradwrtinput):
		#self.bias.grad = gradwrtinput
		#self.weight.grad = gradwrtinput @ 
		return gradwrtinput * self._relu(self.input)

class Sigmoid(Module):

	def __init__(self):
		super(Sigmoid, self).__init__()
		self.input = None

	def forward(self, input):
		self.input = input
		return self._sigmoid(input)

	def _sigmoid(self,x):
		return x.exp()/(x.exp()+1.0)

	def backward(self, gradwrtinput):
		return gradwrtinput * self._sigmoid(self.input) * (1.0 - self._sigmoid(self.input))
