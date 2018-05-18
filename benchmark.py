from torch import Tensor
from torch import nn
import torch.optim as optim
import torch
import math

import matplotlib
import matplotlib.pyplot as plt

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

def plot_points(points, labels):
    colors = ["red", "blue"]
    for i in range(labels.shape[0]):
        c = colors[labels[i]]
        plt.plot(points[i][0], points[i][1], color=c, marker='o')
    plt.show()


batch_size = 1000
train_input, train_target = generate_disc_set(batch_size)
test_input, test_target = generate_disc_set(batch_size)

train_label = conv_to_one_hot(train_target)
test_label = conv_to_one_hot(test_target)

#plot_points(train_input, train_target)

D_in = 2
l1 = 25
l2 = 25
l3 = 25
D_out = 2

lr_rate = 0.1
epochs = 2000

net = nn.Sequential(
        nn.Linear(D_in, l1),
        nn.ReLU(),
        nn.Linear(l1, l2),
        nn.ReLU(),
        nn.Linear(l2, l3),
        nn.ReLU(),
        nn.Linear(l3, D_out),
        nn.Tanh()
    )


criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=lr_rate)

train_loss = []

for epoch in range(epochs):  # loop over the dataset multiple times

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = net(train_input)
    loss = criterion(outputs, train_label)
    loss.backward()
    optimizer.step()

    # print statistics
    print('loss = {}'.format(loss.item()))
    train_loss.append(loss.item())

print('Finished Training')

# Print number of errors
pred_max = predict(net, train_input)
nb_errors = compute_nb_errors(pred_max, train_target)
print('Number of train errors = {} out of {}'.format(nb_errors, batch_size))
#plot_points(train_input, pred_max)

pred_max = predict(net, test_input)
nb_errors = compute_nb_errors(pred_max, test_target)
print('Number of test errors = {} out of {}'.format(nb_errors, batch_size))

# Plot and save loss
fig, ax = plt.subplots()
ax.plot(train_loss)

ax.set(xlabel='iteration (s)', ylabel='Training Loss',
    title='The Loss curve for {} epochs'.format(epochs))
ax.grid()

fig.savefig("loss_epochs={}.png".format(epochs))
plt.show()

