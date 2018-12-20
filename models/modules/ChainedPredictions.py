import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class Identity(nn.Module):
		"""docstring for Identity"""
		def __init__(self):
				super(Identity, self).__init__()

		def forward(self, x):
				return x

class Deception(nn.Module):
		"""docstring for Deception"""
		def __init__(self, hiddenChans):
				super(Deception, self).__init__()
				self.hiddenChans = hiddenChans

				_stack1 = []
				_stack2 = []
				_stack3 = []

				self.start = nn.Conv2d(self.hiddenChans, 32, 1)

				_stack1.append(nn.ConvTranspose2d(32, 32, 2, 2, 0))
				_stack1.append(nn.ConvTranspose2d(32, 32, 4, 2, 1))
				_stack1.append(nn.ConvTranspose2d(32, 32, 6, 2, 2))
				_stack1.append(nn.BatchNorm2d(32))
				self.stack1 = nn.ModuleList(_stack1)

				_stack2.append(nn.ConvTranspose2d(32, 32, 2, 2, 0))
				_stack2.append(nn.ConvTranspose2d(32, 32, 4, 2, 1))
				_stack2.append(nn.ConvTranspose2d(32, 32, 6, 2, 2))
				_stack2.append(nn.BatchNorm2d(32))
				self.stack2 = nn.ModuleList(_stack2)

				self.end = nn.Conv2d(32, 1, 3, 1, 1)

		def forward(self, x):
				x = self.start(x)
				x = self.stack1[0](x) + self.stack1[1](x) + self.stack1[2](x)
				x = self.stack2[0](x) + self.stack2[1](x) + self.stack2[2](x)
				x = self.end(x)
				return x


