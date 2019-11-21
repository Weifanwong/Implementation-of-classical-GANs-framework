import torch
import torch.nn as nn

class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator,self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(
				in_channels=1, 
				out_channels=16,
				kernel_size=3, 
				stride=2, 
				padding=1
				),
			nn.LeakyReLU(0.2,inplace=True),
			nn.Conv2d(
				in_channels=16, 
				out_channels=32,
				kernel_size=3, 
				stride=2, 
				padding=1, 
				),	
			nn.BatchNorm2d(32),	
			nn.LeakyReLU(0.2,inplace=True),
			nn.Conv2d(
				in_channels=32, 
				out_channels=64,
				kernel_size=3, 
				stride=2, 
				padding=1, 
				),
			nn.BatchNorm2d(64),							
			nn.LeakyReLU(0.2,inplace=True),	
			nn.Conv2d(
				in_channels=64, 
				out_channels=128,
				kernel_size=4, 
				stride=1, 
				),
			nn.BatchNorm2d(128),							
			nn.LeakyReLU(0.2,inplace=True),	
			)
		self.Linear = nn.Sequential(nn.Linear(128*1*1,1),nn.Sigmoid())
	def forward(self,x):
		# print(x.shape)		
		x = self.conv(x)
		# print(x.shape)		
		x = x.view(x.size(0),-1)
		# print(x.shape)
		x = self.Linear(x)
		return x
