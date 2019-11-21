import torch
import torch.nn as nn

class Generator(nn.Module):
	def __init__(self,input_size):
		super(Generator,self).__init__()
		self.gen = nn.Sequential(
			nn.Linear(input_size+10,256),
			nn.ReLU(True),
			nn.Linear(256,256),
			nn.ReLU(True),
			nn.Linear(256,784),
			nn.Tanh()
			)
	def forward(self,x,onehot):

		x = self.gen(torch.cat([x,onehot],-1))
		return x
