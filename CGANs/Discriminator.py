import torch
import torch.nn as nn

class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator,self).__init__()
		self.dis = nn.Sequential(
			nn.Linear(784+10,256),
			nn.LeakyReLU(0.02),
			nn.Linear(256,256),
			nn.LeakyReLU(0.02),
			nn.Linear(256,1),
			nn.Sigmoid()
			)
	def forward(self,x,onehot):
		# print(x.shape)
		# print(onehot.shape)
		x = self.dis(torch.cat([x,onehot],-1))
		return x
