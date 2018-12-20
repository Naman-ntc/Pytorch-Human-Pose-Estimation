import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class BnReluConv(nn.Module):
	"""docstring for BnReluConv"""
	def __init__(self, inChannels, outChannels, kernelSize = 1, stride = 1, padding = 0):
		super(BnReluConv, self).__init__()
		self.inChannels = inChannels
		self.outChannels = outChannels
		self.kernelSize = kernelSize
		self.stride = stride
		self.padding = padding

		self.bn = nn.BatchNorm2d(self.inChannels)
		self.relu = nn.ReLU()
		self.conv = nn.Conv2d(self.inChannels, self.outChannels, self.kernelSize, self.stride, self.padding)

	def forward(self, x):
		x = self.bn(x)
		x = self.relu(x)
		x = self.conv(x)
		return x

class Pyramid(nn.Module):
	"""docstring for Pyramid"""
	def __init__(self, D, cardinality, inputRes):
		super(Pyramid, self).__init__()
		self.D = D
		self.cardinality = cardinality
		self.inputRes = inputRes
		self.scale = 2**(-1/self.cardinality)
		_scales = []
		for card in range(self.cardinality):
			temp = nn.Sequential(
					nn.FractionalMaxPool2d(2, output_ratio = self.scale**(card + 1)),
					nn.Conv2d(self.D, self.D, 3, 1, 1),
					nn.Upsample(size = self.inputRes)#, mode='bilinear')
				)
			_scales.append(temp)
		self.scales = nn.ModuleList(_scales)

	def forward(self, x):
		#print(x.shape, self.inputRes)
		out = torch.zeros_like(x)
		for card in range(self.cardinality):
			out += self.scales[card](x)
		return out

class BnReluPyra(nn.Module):
	"""docstring for BnReluPyra"""
	def __init__(self, D, cardinality, inputRes):
		super(BnReluPyra, self).__init__()
		self.D = D
		self.cardinality = cardinality
		self.inputRes = inputRes
		self.bn = nn.BatchNorm2d(self.D)
		self.relu = nn.ReLU()
		self.pyra = Pyramid(self.D, self.cardinality, self.inputRes)

	def forward(self, x):
		x = self.bn(x)
		x = self.relu(x)
		x = self.pyra(x)
		return x


class ConvBlock(nn.Module):
	"""docstring for ConvBlock"""
	def __init__(self, inChannels, outChannels):
		super(ConvBlock, self).__init__()
		self.inChannels = inChannels
		self.outChannels = outChannels
		self.outChannelsby2 = outChannels//2

		self.cbr1 = BnReluConv(self.inChannels, self.outChannelsby2, 1, 1, 0)
		self.cbr2 = BnReluConv(self.outChannelsby2, self.outChannelsby2, 3, 1, 1)
		self.cbr3 = BnReluConv(self.outChannelsby2, self.outChannels, 1, 1, 0)

	def forward(self, x):
		x = self.cbr1(x)
		x = self.cbr2(x)
		x = self.cbr3(x)
		return x

class PyraConvBlock(nn.Module):
	"""docstring for PyraConvBlock"""
	def __init__(self, inChannels, outChannels, inputRes, baseWidth, cardinality, type = 1):
		super(PyraConvBlock, self).__init__()
		self.inChannels = inChannels
		self.outChannels = outChannels
		self.inputRes = inputRes
		self.baseWidth = baseWidth
		self.cardinality = cardinality
		self.outChannelsby2 = outChannels//2
		self.D = self.outChannels // self.baseWidth
		self.branch1 = nn.Sequential(
				BnReluConv(self.inChannels, self.outChannelsby2, 1, 1, 0),
				BnReluConv(self.outChannelsby2, self.outChannelsby2, 3, 1, 1)
			)
		self.branch2 = nn.Sequential(
				BnReluConv(self.inChannels, self.D, 1, 1, 0),
				BnReluPyra(self.D, self.cardinality, self.inputRes),
				BnReluConv(self.D, self.outChannelsby2, 1, 1, 0)
			)
		self.afteradd = BnReluConv(self.outChannelsby2, self.outChannels, 1, 1, 0)

	def forward(self, x):
		x = self.branch2(x) + self.branch1(x)
		x = self.afteradd(x)
		return x

class SkipLayer(nn.Module):
	"""docstring for SkipLayer"""
	def __init__(self, inChannels, outChannels):
		super(SkipLayer, self).__init__()
		self.inChannels = inChannels
		self.outChannels = outChannels
		if (self.inChannels == self.outChannels):
			self.conv = None
		else:
			self.conv = nn.Conv2d(self.inChannels, self.outChannels, 1)

	def forward(self, x):
		if self.conv is not None:
			x = self.conv(x)
		return x

class Residual(nn.Module):
	"""docstring for Residual"""
	def __init__(self, inChannels, outChannels, inputRes=None, baseWidth=None, cardinality=None, type=None):
		super(Residual, self).__init__()
		self.inChannels = inChannels
		self.outChannels = outChannels
		self.cb = ConvBlock(self.inChannels, self.outChannels)
		self.skip = SkipLayer(self.inChannels, self.outChannels)

	def forward(self, x):
		out = 0
		out = out + self.cb(x)
		out = out + self.skip(x)
		return out

class ResidualPyramid(nn.Module):
	"""docstring for ResidualPyramid"""
	def __init__(self, inChannels, outChannels, inputRes, baseWidth, cardinality, type = 1):
		super(ResidualPyramid, self).__init__()
		self.inChannels = inChannels
		self.outChannels = outChannels
		self.inputRes = inputRes
		self.baseWidth = baseWidth
		self.cardinality = cardinality
		self.type = type
		self.cb = PyraConvBlock(self.inChannels, self.outChannels, self.inputRes, self.baseWidth, self.cardinality, self.type)
		self.skip = SkipLayer(self.inChannels, self.outChannels)

	def forward(self, x):
		out = 0
		out = out + self.cb(x)
		out = out + self.skip(x)
		return out
