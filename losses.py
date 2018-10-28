import torch
import torch.nn as nn
import torch.nn.functional as F 

class Loss(object):
	"""docstring for Loss"""
	def __init__(self, opts):
		super(Loss, self).__init__()
		self.opts = opts
	
	def StackedHourGlass(self, output, target):
		loss = 0
		for i in range(self.opts.nStack):
			loss += F.mse_loss(output[i], target)
		return loss

	def ChainedPredictions(self, output, target):
		return F.mse_loss(output, target)

	def DeepPose(self, output, target):
		mask = (target > -0.5 + 1e-8).float()
		return F.mse_loss(output*mask, target*mask)