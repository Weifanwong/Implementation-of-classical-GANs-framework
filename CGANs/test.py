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
fixed_c = torch.FloatTensor(100, 10).zero_()
print(torch.LongTensor(np.array(np.arange(0, 10).tolist()*10).reshape([100, 1])))
fixed_c = fixed_c.scatter_(dim=1, index=torch.LongTensor(np.array(np.arange(0, 10).tolist()*10).reshape([100, 1])), value=1)
# fixed_c = fixed_c.to(device)

