from Discriminator import Discriminator
from Generator import Generator
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import os
import numpy as np

def to_img(x):
    out = 0.5 * (x + 1)
    # out = out.clamp(0, 1)
    out = out.view(-1, 1, 28, 28)
    return out
#hyperparameters
batch_size = 50
num_epoch = 100
input_dimension = 128
#Image processing
img_transform = transforms.Compose([
	transforms.ToTensor(),
	# transforms.Lambda(lambda x: x.repeat(1,1,1)),	
	transforms.Normalize(mean=(0.5,),std=(0.5,))
	])
#mnist datasets
mnist = datasets.MNIST(root='./data/mnist/',train=True,transform=img_transform,download=True)
dataloader = torch.utils.data.DataLoader(dataset=mnist,batch_size=batch_size,shuffle=True)

check_point_dir = 'F:\\pytorch\\CGAN\\checkpoint\\'
try:
	checkpoint = torch.load(check_point_dir+'discriminator.pth')
	Dis = Discriminator()
	Dis.load_state_dict(checkpoint['model_state_dict'])

	checkpoint = torch.load(check_point_dir+'generator.pth')	
	Gen = Generator(input_dimension)
	Gen.load_state_dict(checkpoint['model_state_dict'])
	start_epoch = checkpoint['epoch']
except:
	Dis = Discriminator()
	Gen = Generator(input_dimension)
	start_epoch = 0

criterion = nn.BCELoss()
D_optimizer = torch.optim.Adam(Dis.parameters(),lr=0.0003)
G_optimizer = torch.optim.Adam(Gen.parameters(),lr=0.0003)
ones = Variable(torch.ones(batch_size))
zeros = Variable(torch.zeros(batch_size))

for epoch in range(start_epoch,num_epoch):
	for i, (real_img,real_label) in enumerate(dataloader):
		real_img = real_img.view(batch_size,-1)
		# real_img = to_img(real_img.cpu().data)
		real_onehot = torch.FloatTensor(batch_size, 10).zero_()
		real_onehot = real_onehot.scatter_(dim=1, index=torch.LongTensor(np.array(real_label).reshape([batch_size, 1])), value=1)

		#Discriminator training
		input_noise = Variable(torch.randn(batch_size,input_dimension))
		fake_label = np.array(np.arange(0, 10).tolist()*int(batch_size/10)).reshape([batch_size, 1])
		fake_onehot = torch.FloatTensor(batch_size, 10).zero_()
		fake_onehot = fake_onehot.scatter_(dim=1, index=torch.LongTensor(fake_label), value=1)
		fake_img = Gen(input_noise,fake_onehot)		
		fake_img = fake_img.view(batch_size,-1)
		real_img_logits = Dis(real_img,real_onehot)
		fake_img_logits = Dis(fake_img,fake_onehot)
		D_loss_real = criterion(real_img_logits,ones)
		D_loss_fake = criterion(fake_img_logits,zeros)
		D_loss = D_loss_real + D_loss_fake
		# print(D_loss.data)

		D_optimizer.zero_grad() #
		D_loss.backward()
		D_optimizer.step()

		#Generator training
		input_noise = Variable(torch.randn(batch_size,input_dimension))
		input_noise = Variable(torch.randn(batch_size,input_dimension))
		fake_label = np.array(np.arange(0, 10).tolist()*int(batch_size/10)).reshape([batch_size, 1])
		fake_onehot = torch.FloatTensor(batch_size, 10).zero_()
		fake_onehot = fake_onehot.scatter_(dim=1, index=torch.LongTensor(fake_label), value=1)		
		fake_img = Gen(input_noise,fake_onehot)	
		fake_img = fake_img.view(batch_size,-1)		
		fake_img_logits = Dis(fake_img,fake_onehot)
		G_loss_fake = criterion(fake_img_logits,ones)
		G_loss = G_loss_fake
		G_optimizer.zero_grad()
		G_loss.backward()
		G_optimizer.step()
		if (i + 1) % 1 == 0:
			print('Epoch: {}, ratio: {}/{}, D_real: {:.4f}, D_fake: {:.4f}'.format(epoch,i,len(dataloader),real_img_logits.data.mean(),fake_img_logits.data.mean()))
			fake_img = to_img(fake_img.cpu().data)
			save_image(fake_img,'./img/fake_images-{}.png'.format(epoch + 1))

torch.save(Gen.state_dict(),'./generator.pth')
torch.save(Dis.state_dict(),'./discriminator.pth')