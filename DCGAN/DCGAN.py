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
import torch.autograd as autograd


def to_img(x):
    out = x
    # out = 0.5 * (x + 1)
    out = out.clamp(0, 1)
    out = out.view(-1, 1, 32, 32)
    return out
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

#hyperparameters
batch_size = 64
num_epoch = 100
input_dimension = 100
# weight_cliping_limit = 0.01
#Image processing
img_transform = transforms.Compose([
	transforms.Resize(32),
	transforms.ToTensor(),
	# transforms.Lambda(lambda x: x.repeat(3,1,1)),		
	transforms.Normalize(mean=(0.5,),std=(0.5,))
	])
# #mnist datasets
# mnist = datasets.MNIST(root='./data/mnist/',train=True,transform=img_transform,download=True)
# #split batch
# dataloader = torch.utils.data.DataLoader(dataset=mnist,batch_size=batch_size,shuffle=True)

transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
dataset = torchvision.datasets.MNIST(root='data/', train=True, transform=transform, download=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
#
check_point_dir = 'F:\\pytorch\\DCGAN\\checkpoint\\'
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
	Dis.apply(weights_init_normal)
	Gen.apply(weights_init_normal)	
	start_epoch = 0
criterion = nn.BCELoss()	
D_optimizer = torch.optim.Adam(Dis.parameters(),lr=0.0002)
G_optimizer = torch.optim.Adam(Gen.parameters(),lr=0.0002)

for epoch in range(start_epoch,num_epoch):
	for i, (img,_) in enumerate(dataloader):
		img = img.view(batch_size,-1)
		# real_img = img.to(torch.device('cpu'))
		real_img = to_img(img.cpu().data)
		real_label = Variable(torch.ones(real_img.shape[0]))
		fake_label = Variable(torch.zeros(real_img.shape[0]))
		# print(real_img.shape)

		#Discriminator training
		input_noise = Variable(torch.randn(real_img.shape[0],input_dimension))
		fake_img = Gen(input_noise)
		# print(real_img.shape)
		real_img_logits = Dis(real_img)
		fake_img_logits = Dis(fake_img)

		#D loss for GANs
		D_loss_real = criterion(real_img_logits,real_label)
		D_loss_fake = criterion(fake_img_logits,fake_label)
		D_loss = D_loss_real + D_loss_fake
		D_optimizer.zero_grad() #
		D_loss.backward()
		D_optimizer.step()

		#clip for D parameters
		# for p in Dis.parameters():
		# 	p.data.clamp_(-weight_cliping_limit, weight_cliping_limit)

		#Generator training
		input_noise = Variable(torch.randn(real_img.shape[0],input_dimension))	
		fake_img = Gen(input_noise)	
		fake_img_logits = Dis(fake_img)

		#G loss for GANs
		G_loss_fake = criterion(fake_img_logits,real_label)
		G_loss = G_loss_fake
		G_optimizer.zero_grad()
		G_loss.backward()
		G_optimizer.step()
		if (i + 1) % 1 == 0:
			print('Epoch: {}, ratio: {}/{}, D_real: {:.4f}, D_fake: {:.4f}'.format(epoch,i,len(dataloader),real_img_logits.data.mean(),fake_img_logits.data.mean()))

	fake_img = to_img(fake_img.cpu().data)
	save_image(fake_img,'./img/fake_images-{}.png'.format(epoch + 1))

	torch.save({
		'epoch':epoch,
		'model_state_dict':Gen.state_dict(),
		'optimizer_state_dict':G_optimizer.state_dict(),
		},
		'./checkpoint/generator.pth')
	torch.save({
		'epoch':epoch,
		'model_state_dict':Dis.state_dict(),
		'optimizer_state_dict':D_optimizer.state_dict(),
		},
		'./checkpoint/discriminator.pth')