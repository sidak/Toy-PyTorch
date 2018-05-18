from torch import Tensor
from torch import nn
import torch.optim as optim
import torch
import math

import matplotlib
import matplotlib.pyplot as plt
from activation import Relu, Tanh

def generate_disc_set(nb):
    input = Tensor(nb, 2).uniform_(0, 1)
    modified_input = input - 0.5
    #print(modified_input.pow(2).sum(1).sub(1/ (2*math.pi)).sign().mul(-1))
    #print("Modified inputs are ", modified_input)
    target = modified_input.pow(2).sum(dim=1).sub(1/ (2*math.pi)).sign().mul(-1).add(1).div(2).long()
    return input, target

def conv_to_one_hot(labels):
    one_hot = torch.Tensor(labels.shape[0], labels.max()+1).zero_()
    one_hot.scatter_(1, labels.view(-1, 1), 1.0)
    return one_hot

def compute_nb_errors(pred, tgt):
    return (pred!=tgt).long().sum()

def predict(net, input):
    pred = net(input)

    pred_t = pred.t()
    pred_max = pred_t.max(0)[1]

    return pred_max


batch_size = 1000
train_input, train_target = generate_disc_set(batch_size)
test_input, test_target = generate_disc_set(batch_size)

train_label = conv_to_one_hot(train_target)
test_label = conv_to_one_hot(test_target)


D_in = 2
l1 = 100
l2 = 200
l3 = 50
D_out = 2

lr_rate = 1e-3
epochs = 100

net = nn.Sequential(
        nn.Linear(D_in, l1),
        nn.ReLU(),
        nn.Linear(l1, l2),
        nn.ReLU(),
        nn.Linear(l2, l3),
        nn.ReLU(),
        nn.Linear(l3, D_out),
        nn.ReLU()
    )


criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=lr_rate)


for epoch in range(epochs):  # loop over the dataset multiple times

    running_loss = 0.0

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = net(train_input)
    loss = criterion(outputs, train_label)
    loss.backward()
    optimizer.step()

    # print statistics
    print('loss = {}'.format(loss.item()))

print('Finished Training')

pred_max = predict(net, train_input)
nb_errors = compute_nb_errors(pred_max, train_target)
print('Number of train errors = {} out of {}'.format(nb_errors, batch_size))

pred_max = predict(net, test_input)
nb_errors = compute_nb_errors(pred_max, test_target)
print('Number of test errors = {} out of {}'.format(nb_errors, batch_size))
