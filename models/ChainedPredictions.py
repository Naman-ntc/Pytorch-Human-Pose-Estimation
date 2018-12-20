import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import models.modules.ChainedPredictions as M

class ChainedPredictions(nn.Module):
	"""docstring for ChainedPredictions"""
	def __init__(self, modelName, hhKernel, ohKernel, nJoints):
		super(ChainedPredictions, self).__init__()
		self.nJoints = nJoints
		self.modelName = modelName
		self.resnet = getattr(torchvision.models, self.modelName)(pretrained=True)
		self.resnet.avgpool = M.Identity()
		self.resnet.fc = M.Identity()
		self.hiddenChans = 64 ### Add cases!

		self.hhKernel = hhKernel
		self.ohKernel = ohKernel

		self.init_hidden = nn.Conv2d(512, self.hiddenChans, 1)
		_deception = []
		for i in range(self.nJoints):
			_deception.append(M.Deception(self.hiddenChans))
		self.deception = nn.ModuleList(_deception)

		_h2h = []
		_o2h = []
		for i in range(nJoints):
			_o = []
			_h2h.append(
				nn.Sequential(
					nn.Conv2d(self.hiddenChans, self.hiddenChans, kernel_size=self.hhKernel, padding=self.hhKernel//2),
					nn.BatchNorm2d(self.hiddenChans)
				)
			)
			for j in range(i+1):
				_o.append(nn.Sequential(
						nn.Conv2d(1, self.hiddenChans, 1),
						nn.Conv2d(self.hiddenChans, self.hiddenChans, kernel_size=self.ohKernel, stride=2, padding=self.ohKernel//2),
						nn.BatchNorm2d(self.hiddenChans),
						nn.Conv2d(self.hiddenChans, self.hiddenChans, kernel_size=self.ohKernel, stride=2, padding=self.ohKernel//2),
						nn.BatchNorm2d(self.hiddenChans),
					)
				)
			_o2h.append(nn.ModuleList(_o))

		self.h2h = nn.ModuleList(_h2h)
		self.o2h = nn.ModuleList(_o2h)

	def forward(self, x):
		hidden = [0]*self.nJoints
		output = [None]*self.nJoints
		hidden[0] += self.resnet(x).reshape(-1, 512, 8, 8)
		hidden[0] = self.init_hidden(hidden[0])
		output[0] = self.deception[0](hidden[0])

		for i in range(self.nJoints-1):
			hidden[i+1] = self.h2h[i](hidden[i])
			for j in range(i+1):
				hidden[i+1] += self.o2h[i][j](output[j])
			hidden[i+1] = torch.relu(hidden[i+1])
			output[i+1] = self.deception[i+1](hidden[i+1])
		return torch.cat(output, 1)
