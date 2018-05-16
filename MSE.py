from module import Module
from variable import Variable
from torch import Tensor
from linear import Linear

class MSE(Module):
	def __init__(self):
		super(MSE, self).__init__()

	def forward(self, pred, y):	 
		return (pred.data - y.data).pow(2).sum()/y.shape[0]

	def backward(self, pred, y):
		return 2*(pred.data - y.data).sum()/y.shape[0]

if __name__ == '__main__':
	x = Variable(Tensor([1, 2, 3]))
	y = Variable(Tensor([8,2]))
	net = Linear(3,2)
	pred = Variable(Tensor([0,1]))
	lossfunc = MSE()
	out = lossfunc.forward(pred,y)
	print(out)