import nnlib
from torch import Tensor

x = torch.Tensor([1, 2, 3])
y = torch.Tensor([7])

linear = nnlib.Linear(x.shape[0], y.shape[0])

loss = nnlib.MSE(linear, y)

loss.backward()

print(x.grad)


