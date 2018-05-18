from module import Module
from torch import Tensor
from torch import FloatTensor

from linear import Linear

class MSE(Module):
	
	def __init__(self):
		super(MSE, self).__init__()
		self.pred = None
		self.y = None
	def forward(self, pred, y):	 
		self.pred = pred
		self.y = y
		return (pred - y).pow(2).sum(dim=0).sum(dim=0)/y.shape[1]

	def backward(self):
		return 2*(self.pred - self.y)/self.y.shape[1]