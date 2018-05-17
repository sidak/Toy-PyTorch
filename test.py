from linear import Linear
from torch import Tensor
from network import Network
from MSE import MSE
import matplotlib
import matplotlib.pyplot as plt

def run_mini_example():

	x = Tensor([1, 2, 3])
	y = Tensor([7, 10])
	print(x.shape, y.shape)

	linear = Linear(x.shape[0], y.shape[0], weight_init='ones')
	net = Network([linear])

	pred = net.forward(x)


	
	#loss.backward()
	print("Pred is ")
	print(pred)
	#print(x.grad)

def run_bigger_example():
	x = Tensor([1, 2, 3])
	y = Tensor([7, 10])
	print(x.shape, y.shape)

	linear1 = Linear(x.shape[0], x.shape[0], weight_init='ones')
	linear2 = Linear(x.shape[0], y.shape[0], weight_init='ones')
	
	net_2layer = Network([linear1, linear2])
	pred_2layer = net_2layer.forward(x)

	#loss.backward()
	print("pred_2layer is ")
	print(pred_2layer)	
	mse = MSE()
	loss = mse.forward(pred_2layer, y)
	print("loss for 2 layer net is ")
	print(loss)
	# Should be 2*(18-7) = 22
	loss_grad = mse.backward()
	print("loss_grad for 2layer net is ")
	print(loss_grad)
	print("Printing params Grad before ")
	for layer in net_2layer.layers:
		for par_grad in layer.param_grad():
			print(par_grad)

	print("now setting param grad to zero")
	net_2layer.zero_grad()
	print("Printing params Grad after ")
	for layer in net_2layer.layers:
		for par_grad in layer.param_grad():
			print(par_grad)
	print("Printing params before backward")
	for layer in net_2layer.layers:
		for par in layer.param():
			print(par)
	print("Doing backward pass")
	net_2layer.backward(loss_grad)
	print("Printing params after backward")
	for layer in net_2layer.layers:
		for par in layer.param():
			print(par)
	print("Printing params Grad")
	for layer in net_2layer.layers:
		for par_grad in layer.param_grad():
			print(par_grad)
	print("Doing param update")
	net_2layer.grad_step(lr=1e-3)
	print("Printing params after update")
	for layer in net_2layer.layers:
		for par in layer.param():
			print(par)


if __name__ == '__main__':

	x = Tensor([1, 2, 3])
	y = Tensor([7, 10])
	print(x.shape, y.shape)

	linear1 = Linear(x.shape[0], x.shape[0], weight_init='ones')
	linear2 = Linear(x.shape[0], y.shape[0], weight_init='ones')
	
	net_2layer = Network([linear1, linear2])
	mse = MSE()

	lr = 1e-3
	num_iter = 200

	timesteps = []
	loss_at_timesteps = []

	for it in range(num_iter):
		
		net_2layer.zero_grad()
		pred_2layer = net_2layer.forward(x)
		loss = mse.forward(pred_2layer, y)
		print("At iteration ", str(it), " the loss is ", loss)
		loss_grad = mse.backward()
		net_2layer.backward(loss_grad)
		net_2layer.grad_step(lr=1e-3)

		timesteps.append(it)
		loss_at_timesteps.append(loss)

	print("Prediction at the end ", net_2layer.forward(x))

	fig, ax = plt.subplots()
	ax.plot(timesteps, loss_at_timesteps)

	ax.set(xlabel='iteration (s)', ylabel='Training Loss',
	       title='The Loss curve')
	ax.grid()

	#fig.savefig("test.png")
	plt.show()
