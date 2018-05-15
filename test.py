from variable import Variable
from linear import Linear
from mse import MSE

if __name__ == '__main__':

	x = Variable([1, 2, 3])
	y = Variable([7])

	linear = Linear(x.shape[0], y.shape[0])

	loss = MSE(linear, y)

	loss.backward()

	print(x.grad)


