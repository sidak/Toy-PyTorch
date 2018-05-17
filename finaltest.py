from torch import Tensor
import math


def generate_disc_set(nb):
    input = Tensor(nb, 2).uniform_(0, 1)
    print(input.pow(2).sum(1).sub(1/ (2*math.pi)).sign())
    target = input.pow(2).sum(1).sub(1/ (2*math.pi)).sign().mul(-1).add(1).div(2).long()
    return input, target

train_input, train_target = generate_disc_set(10)
#test_input, test_target = generate_disc_set(1000)


print(train_input, train_target)
'''
mean, std = train_input.mean(), train_input.std()

train_input.sub_(mean).div_(std)
test_input.sub_(mean).div_(std)

train_input, train_target = Variable(train_input), Variable(train_target)
test_input, test_target = Variable(test_input), Variable(test_target)

mini_batch_size = 100
'''