from variable import Variable
from linear import Linear
from torch import Tensor
from network import Network
from MSE import MSE


if __name__ == '__main__':

	x = Variable(Tensor([1, 2, 3]))
	y = Variable(Tensor([7]))
	print(x.shape, y.shape)

	linear = Linear(x.shape[0], y.shape[0], weight_init='ones')
	net = Network([linear])

	pred = net.forward(x)


	
	#loss.backward()
	print("Pred is ")
	print(pred)
	#print(x.grad)

	linear1 = Linear(x.shape[0], x.shape[0], weight_init='ones')
	linear2 = Linear(x.shape[0], y.shape[0], weight_init='ones')
	
	net_2layer = Network([linear1, linear2])

	pred_2layer = net_2layer.forward(x)

	#loss.backward()
	print("pred_2layer is ")
	print(pred_2layer)
	
	mse = MSE()
	loss = mse.forward(pred_2layer, y)
	print("loss is ")
	print(loss)

	
