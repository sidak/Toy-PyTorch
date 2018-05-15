from variable import Variable
from linear import Linear
from torch import Tensor
#from mse import MSE


if __name__ == '__main__':

	x = Variable(Tensor([1, 2, 3]))
	y = Variable(Tensor([7]))
	print(x.shape, y.shape)
	linear = Linear(x.shape[0], y.shape[0])
	pred = linear.forward(x)

	#loss = MSE(linear, y)

	#loss.backward()
	print("Pred is ")
	print(pred)
	#print(x.grad)
