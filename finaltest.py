from torch import Tensor
import math
from linear import Linear
from torch import Tensor
from network import Network
from MSE import MSE
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

train_input, train_target = generate_disc_set(1000)
test_input, test_target = generate_disc_set(1000)

num_hidden = 3

layers = []
linear = Linear(2, 25, weight_init='ones')
layers.append(linear)
layers.append(Relu())
for i in range(num_hidden-1):
	layers.append(Linear(25, 25, weight_init='ones'))
	layers.append(Relu())
layers.append(Linear(25, 2, weight_init='ones'))


net_2layer = Network(layers, train_input.shape[0])

mse = MSE()


lr = 1e-3
num_iter = 20000

timesteps = []
loss_at_timesteps = []

for it in range(num_iter):
	
	net_2layer.zero_grad()
	pred_2layer = net_2layer.forward(train_input.view(-1, train_input.shape[0]))

	loss = mse.forward(pred_2layer,test_input.view(-1, test_input.shape[0]))
	print(loss)
	#print("At iteration ", str(it), " the loss is ", loss)
	loss_grad = mse.backward()
	net_2layer.backward(loss_grad)
	net_2layer.grad_step(lr=1e-3)

	timesteps.append(it)
	loss_at_timesteps.append(loss)

pred = net_2layer.forward(train_input.view(-1, train_input.shape[0]))
print("Prediction at the end ", pred)



# fig, ax = plt.subplots()
# ax.plot(timesteps, loss_at_timesteps)

# ax.set(xlabel='iteration (s)', ylabel='Training Loss',
#        title='The Loss curve')
# ax.grid()

# #fig.savefig("test.png")
# plt.show()

