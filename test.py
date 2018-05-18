from torch import Tensor
from random import shuffle
import math

from linear import Linear
from network import Network
from MSE import MSE
from activation import Relu, Tanh


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
    one_hot = Tensor(labels.shape[0], 2).zero_()
    one_hot.scatter_(1, labels.view(-1, 1), 1.0)
    return one_hot

def compute_nb_errors(pred, tgt):

    return (pred!=tgt).long().sum()


# Generate Data

train_input, train_target = generate_disc_set(1000)
test_input, test_target = generate_disc_set(1000)

train_target_hot = conv_to_one_hot(train_target)

# Build network

num_hidden = 3
weight_init ='pytorch_default' 
bias_init = 'zero' 
layers = []


linear = Linear(2, 25, weight_init=weight_init, bias_init=bias_init)
layers.append(linear)
layers.append(Relu())
for i in range(num_hidden-1):
    layers.append(Linear(25, 25, weight_init=weight_init, bias_init=bias_init))
    layers.append(Relu())
layers.append(Linear(25, 2, weight_init=weight_init, bias_init=bias_init))
layers.append(Tanh())
net_2layer = Network(layers, train_input.shape[0])

# Choose loss

mse = MSE()

# Choose parameters

lr = 0.05
num_iter = 1000

timesteps = []
loss_at_timesteps = []

# Train model

for it in range(num_iter):

    net_2layer.zero_grad()

    pred_2layer = net_2layer.forward(train_input.t())
    
    loss = mse.forward(pred_2layer, train_target_hot.t())

    print("At iteration ", str(it), " the loss is ", loss)
    loss_grad = mse.backward()
    net_2layer.backward(loss_grad)

    net_2layer.grad_step(lr=lr)

    timesteps.append(it)
    loss_at_timesteps.append(loss)


final_pred_train = net_2layer.forward(train_input.t())
print('Number of training errors:')
print(compute_nb_errors(final_pred_train.max(0)[1], train_target))

final_pred_test = net_2layer.forward(test_input.t())
print('Number of test errors:')
print(compute_nb_errors(final_pred_test.max(0)[1], test_target))
