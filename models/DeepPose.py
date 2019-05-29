import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import models.modules.DeepPose as M

class DeepPose(nn.Module):
	"""docstring for DeepPose"""
	def __init__(self, nJoints, modelName='resnet50'):
		super(DeepPose, self).__init__()
		self.nJoints = nJoints
		self.block = 'BottleNeck' if (int(modelName[6:]) > 34) else 'BasicBlock'
		self.resnet = getattr(torchvision.models, modelName)(pretrained=True)
		self.resnet.fc = nn.Linear(512 * (4 if self.block == 'BottleNeck' else 1), self.nJoints * 2)
	def forward(self, x):
		return self.resnet(x)
