from torch import Tensor
import torch
import math
from linear import Linear
from torch import Tensor
from network import Network
from MSE import MSE
import matplotlib
import matplotlib.pyplot as plt
from activation import Relu, Tanh
from random import shuffle

def generate_disc_set(nb):
    input = Tensor(nb, 2).uniform_(0, 1)
    modified_input = input - 0.5
    #print(modified_input.pow(2).sum(1).sub(1/ (2*math.pi)).sign().mul(-1))
    #print("Modified inputs are ", modified_input)
    target = modified_input.pow(2).sum(dim=1).sub(1/ (2*math.pi)).sign().mul(-1).add(1).div(2).long()
    return input, target

def generate_disc_set_origin(nb):
    input = Tensor(nb, 2).uniform_(0, 1)
    #print(modified_input.pow(2).sum(1).sub(1/ (2*math.pi)).sign().mul(-1))
    #print("Modified inputs are ", modified_input)
    target = input.pow(2).sum(dim=1).sub(1/ (2*math.pi)).sign().mul(-1).add(1).div(2).long()
    return input, target


def conv_to_one_hot(labels):
	one_hot = torch.Tensor(labels.shape[0], 2).zero_()
	one_hot.scatter_(1, labels.view(-1, 1), 1.0)
	return one_hot

def compute_nb_errors(pred, tgt):
	return (pred!=tgt).long().sum()

def plot_points(points, labels):
	colors = ["red", "blue"]
	for i in range(labels.shape[0]):
		c = colors[labels[i]]
		plt.plot(points[i][0], points[i][1], color=c, marker='o')
	plt.show()

train_input, train_target = generate_disc_set(1000)
test_input, test_target = generate_disc_set(1000)

train_target_hot = conv_to_one_hot(train_target)

#plot_points(train_input, train_target)

#print(train_input)
#print(train_target)
#print(train_target_hot)

#num_hidden = 3
num_hidden = 3
weight_init ='pytorch_default' #'uniform_pos_neg'
bias_init = 'zero' 
#bias_init='zero'
layers = []
'''
linear = Linear(2, 25, weight_init=weight_init, bias_init=bias_init)
layers.append(linear)
layers.append(Tanh())
for i in range(num_hidden-1):
	layers.append(Linear(25, 25, weight_init=weight_init, bias_init=bias_init))
	layers.append(Tanh())
layers.append(Linear(25, 2, weight_init=weight_init, bias_init=bias_init))
'''


linear = Linear(2, 25, weight_init=weight_init, bias_init=bias_init)
layers.append(linear)
layers.append(Relu())
for i in range(num_hidden-1):
	layers.append(Linear(25, 25, weight_init=weight_init, bias_init=bias_init))
	layers.append(Relu())
layers.append(Linear(25, 2, weight_init=weight_init, bias_init=bias_init))
layers.append(Tanh())
net_2layer = Network(layers, train_input.shape[0])

mse = MSE()

for layer in net_2layer.layers:
	print([par for par in layer.param()])

lr = 0.05
num_iter = 1000

timesteps = []
loss_at_timesteps = []

for it in range(num_iter):

	#print("The x and y are ")
	#print(x, y)
	net_2layer.zero_grad()
	''' 
	for layer in net_2layer.layers:
		print([par.size() for par in layer.param()])
	'''
	pred_2layer = net_2layer.forward(train_input.t())
	
	
	#print("shape of pred layer", pred_2layer.shape)
	loss = mse.forward(pred_2layer, train_target_hot.t())

	print("At iteration ", str(it), " the loss is ", loss)
	loss_grad = mse.backward()
	net_2layer.backward(loss_grad)

	'''
	print("parameters before applying gradient step")
	for layer in net_2layer.layers:
		print([par for par in layer.param()])
		continue
	'''
	net_2layer.grad_step(lr=lr)
	'''
	print("parameters after applying gradient step")
	for layer in net_2layer.layers:
		print([par for par in layer.param()])
		continue
	
	print("parameter gradient ")
	for layer in net_2layer.layers:
		print([par for par in layer.param_grad()])
		continue
	'''
	timesteps.append(it)
	loss_at_timesteps.append(loss)
'''

for it in range(20):

	#train_input, train_target_hot = shuffle(train_input, train_target_hot)

	for x, y in zip(train_input, train_target_hot) :
		x = x.view(-1, 1)
		y = y.view(-1, 1)
		print("The x and y are ")
		print(x, y)
		net_2layer.zero_grad()

		for layer in net_2layer.layers:
			print([par.size() for par in layer.param()])

		pred_2layer = net_2layer.forward(x)
		
		
		#print("shape of pred layer", pred_2layer.shape)
		loss = mse.forward(pred_2layer, y)
		print("At iteration ", str(it), " the loss is ", loss)
		loss_grad = mse.backward()
		net_2layer.backward(loss_grad)
		net_2layer.grad_step(lr=lr)
		timesteps.append(it)
		loss_at_timesteps.append(loss)

'''

final_pred_train = net_2layer.forward(train_input.t())
print(final_pred_train)
print(final_pred_train.max(0)[1])
print(train_target)
print('Number of training errors:')
print(compute_nb_errors(final_pred_train.max(0)[1], train_target))


final_pred_test = net_2layer.forward(test_input.t())
print('Number of test errors:')
print(compute_nb_errors(final_pred_test.max(0)[1], test_target))
plot_points(test_input, final_pred_test.max(0)[1])
'''

fig, ax = plt.subplots()
ax.plot(timesteps, loss_at_timesteps)

ax.set(xlabel='iteration (s)', ylabel='Training Loss',
	title='The Loss curve')
ax.grid()

fig.savefig("test.png")
plt.show()



for it in range(num_iter):
	for x, y in zip(train_input, train_target_hot) :
		x = x.view(-1, 1)
		y = y.view(-1, 1)
		#print("The x and y are ")
		#print(x, y)
		net_2layer.zero_grad()

		for layer in net_2layer.layers:
			print([par.size() for par in layer.param()])

		pred_2layer = net_2layer.forward(x)
		
		
		#print("shape of pred layer", pred_2layer.shape)
		loss = mse.forward(pred_2layer, y)
		print("At iteration ", str(it), " the loss is ", loss)
		loss_grad = mse.backward()
		net_2layer.backward(loss_grad)
		net_2layer.grad_step(lr=lr)
		timesteps.append(it)
		loss_at_timesteps.append(loss)

'''