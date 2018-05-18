'''
	Interface for the layers (activations, linear)

'''
class Module(object):

	def __init__(self):
		pass

	def forward(self , * input):
		raise NotImplementedError
	
	def backward(self , * gradwrtoutput):
		raise NotImplementedError
	
	def param(self):
		return []

	def param_grad(self):
		return []

	def set_param_grad(self, param_grad):
		pass

	def update_param(self, lr, wd):
		pass

	def init_weights(self, nb_samples):
		pass

	def set_zero_grad(self):
		pass