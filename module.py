class Module(object):

	def __init__(self):
		pass

	def forward(self , * input):
		raise NotImplementedError
	
	def backward(self , * gradwrtoutput):
		raise NotImplementedError
	
	def param(self):
		return []