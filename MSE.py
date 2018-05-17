from module import Module
from variable import Variable
from torch import Tensor
from torch import FloatTensor

from linear import Linear

class MSE(Module):
	def __init__(self):
		super(MSE, self).__init__()

	def forward(self, pred, y):	 
		# dim = 0 is necessary, otherwise returns just a scalar
		return (pred.data - y.data).pow(2).sum(dim=0)/y.shape[0]

	def backward(self, pred, y):
		return 2*(pred.data - y.data).sum(dim=0)/y.shape[0]

if __name__ == '__main__':
	x = Variable(Tensor([1, 2, 3]))
	y = Variable(Tensor([8,2]))
	net = Linear(3,2)
	pred = Variable(Tensor([0,1]))
	lossfunc = MSE()
	out = lossfunc.forward(pred,y)
	print(out)