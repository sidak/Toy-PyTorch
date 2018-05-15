from nnlib import Module

class Activation(Module):

	def __init__():
		
	def forward(self , * input):
		raise NotImplementedError
	
	def backward(self , * gradwrtoutput):
		raise NotImplementedError
	
	def param(self):
		return []