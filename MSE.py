from module import Module
from torch import Tensor

class MSE(Module):
	def __init__(self):
		super(MSE, self).__init__()

	def forward(self, pred, y):
		diff = pred.data - y.data 		
		loss = (diff.view(-1,1) @ diff)/2
		return loss

	def backward():
		raise NotImplementedError
