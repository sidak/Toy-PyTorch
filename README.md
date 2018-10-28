# Toy-PyTorch

This is an implementation a mini deep learning framework from scratch by relying only on `torch.Tensor`.

The API is essentially similar to that of PyTorch, but provides for all the implementations in a simple and concise manner. 

`test.py` illustrates how to use this mini-framework for the example for classifying points lying inside or outside a disc. 

## Design Choices

We follow the modular structure and have an abstract class called `Module`.  All of the other classes thus follow the similar method structure, the concrete definition of which is implemented in the respective class. Most importantly  `forward(self, * input)` and the `backward(self, * gradwrtoutput)` functions in these Modules are later called upon to compute the forward and backward propagation. The modules currently supported are:

- Network,
- Linear (fully connected layers),
- Activation functions (ReLU, Tanh, Sigmoid, Leaky ReLU) and the 
- MSE loss.

The core class that stitches everything together is the `Network`. It takes in a list of `Module`'s which we term as layers. We just have to call upon the Network for doing the forward pass, backward pass and updating the gradients. It internally invokes the forward or backward for each layer. The details of forward and backward are abstracted in the Module classes themselves which provides for a neat object-oriented manner.

While PyTorch used to follow the Variable, we don't keep an extra class for it, as it allows for unnecessary burden of providing same method for Tensor and Variable. This is even evident from change of choice in the latest PyTorch version! 

Since the library is meant to be used solely for the purpose of simple MLP, we opted for simplicity and added the optimizer directly to the network.

This work was done jointly with [Ignacio Aleman](https://github.com/Nacho114) and [Nawel Naas](https://github.com/naweln). 
