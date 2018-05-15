from module import Module
from variable import Variable
from torch import Tensor
from linear import Linear

class MSE(Module):
	def __init__(self):
		super(MSE, self).__init__()

	def forward(self, pred, y):	 
		return (pred.data - y.data).pow(2).sum()/2

	def backward():
		raise NotImplementedError

