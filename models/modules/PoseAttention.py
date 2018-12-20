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
		self.conv = nn.Conv2d(self.inChannels, self.outChannels, self.kernelSize, self.stride, self.padding)
		self.relu = nn.ReLU()

	def forward(self, x):
		x = self.bn(x)
		x = self.relu(x)
		x = self.conv(x)
		return x

class BnReluPoolConv(nn.Module):
		"""docstring for BnReluPoolConv"""
		def __init__(self, inChannels, outChannels, kernelSize = 1, stride = 1, padding = 0):
			super(BnReluPoolConv, self).__init__()
			self.inChannels = inChannels
			self.outChannels = outChannels
			self.kernelSize = kernelSize
			self.stride = stride
			self.padding = padding

			self.bn = nn.BatchNorm2d(self.inChannels)
			self.conv = nn.Conv2d(self.inChannels, self.outChannels, self.kernelSize, self.stride, self.padding)
			self.relu = nn.ReLU()

		def forward(self, x):
			x = self.bn(x)
			x = self.relu(x)
			x = F.max_pool2d(x, kernel_size=2, stride=2)
			x = self.conv(x)
			return x

class ConvBlock(nn.Module):
	"""docstring for ConvBlock"""
	def __init__(self, inChannels, outChannels):
		super(ConvBlock, self).__init__()
		self.inChannels = inChannels
		self.outChannels = outChannels
		self.outChannelsby2 = outChannels//2

		self.brc1 = BnReluConv(self.inChannels, self.outChannelsby2, 1, 1, 0)
		self.brc2 = BnReluConv(self.outChannelsby2, self.outChannelsby2, 3, 1, 1)
		self.brc3 = BnReluConv(self.outChannelsby2, self.outChannels, 1, 1, 0)

	def forward(self, x):
		x = self.brc1(x)
		x = self.brc2(x)
		x = self.brc3(x)
		return x

class PoolConvBlock(nn.Module):
	"""docstring for PoolConvBlock"""
	def __init__(self, inChannels, outChannels):
		super(PoolConvBlock, self).__init__()
		self.inChannels = inChannels
		self.outChannels = outChannels

		self.brpc = BnReluPoolConv(self.inChannels, self.outChannels, 3, 1, 1)
		self.brc = BnReluConv(self.outChannels, self.outChannels, 3, 1, 1)

	def forward(self, x):
		x = self.brpc(x)
		x = self.brc(x)
		x = F.interpolate(x, scale_factor=2)
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
	def __init__(self, inChannels, outChannels):
		super(Residual, self).__init__()
		self.inChannels = inChannels
		self.outChannels = outChannels
		self.cb = ConvBlock(inChannels, outChannels)
		self.skip = SkipLayer(inChannels, outChannels)

	def forward(self, x):
		out = 0
		out = out + self.cb(x)
		out = out + self.skip(x)
		return out

class HourGlassResidual(nn.Module):
	"""docstring for HourGlassResidual"""
	def __init__(self, inChannels, outChannels):
		super(HourGlassResidual, self).__init__()
		self.inChannels = inChannels
		self.outChannels = outChannels
		self.cb = ConvBlock(inChannels, outChannels)
		self.pcb = PoolConvBlock(inChannels, outChannels)
		self.skip = SkipLayer(inChannels, outChannels)


	def forward(self, x):
		out = 0
		out = out + self.cb(x)
		out = out + self.pcb(x)
		out = out + self.skip(x)
		return out

class AttentionIter(nn.Module):
	"""docstring for AttentionIter"""
	def __init__(self, nChannels, LRNSize, IterSize):
		super(AttentionIter, self).__init__()
		self.nChannels = nChannels
		self.LRNSize = LRNSize
		self.IterSize = IterSize
		self.bn = nn.BatchNorm2d(self.nChannels)
		self.U = nn.Conv2d(self.nChannels, 1, 3, 1, 1)
		# self.spConv = nn.Conv2d(1, 1, self.LRNSize, 1, self.LRNSize//2)
		# self.spConvclone = nn.Conv2d(1, 1, self.LRNSize, 1, self.LRNSize//2)
		# self.spConvclone.load_state_dict(self.spConv.state_dict())
		_spConv_ = nn.Conv2d(1, 1, self.LRNSize, 1, self.LRNSize//2)
		_spConv = []
		for i in range(self.IterSize):
			_temp_ = nn.Conv2d(1, 1, self.LRNSize, 1, self.LRNSize//2)
			_temp_.load_state_dict(_spConv_.state_dict())
			_spConv.append(nn.BatchNorm2d(1))
			_spConv.append(_temp_)
		self.spConv = nn.ModuleList(_spConv)

	def forward(self, x):
		x = self.bn(x)
		u = self.U(x)
		out = u
		for i in range(self.IterSize):
			# if (i==1):
			# 	out = self.spConv(out)
			# else:
			# 	out = self.spConvclone(out)
			out = self.spConv[2*i](out)
			out = self.spConv[2*i+1](out)
			out = u + torch.sigmoid(out)
		return (x * out.expand_as(x))

class AttentionPartsCRF(nn.Module):
	"""docstring for AttentionPartsCRF"""
	def __init__(self, nChannels, LRNSize, IterSize, nJoints):
		super(AttentionPartsCRF, self).__init__()
		self.nChannels = nChannels
		self.LRNSize = LRNSize
		self.IterSize = IterSize
		self.nJoints = nJoints
		_S = []
		for _ in range(self.nJoints):
			_S_ = []
			_S_.append(AttentionIter(self.nChannels, self.LRNSize, self.IterSize))
			_S_.append(nn.BatchNorm2d(self.nChannels))
			_S_.append(nn.Conv2d(self.nChannels, 1, 1, 1, 0))
			_S.append(nn.Sequential(*_S_))
		self.S = nn.ModuleList(_S)

	def forward(self, x):
		out = []
		for i in range(self.nJoints):
			#out.append(self.S[i](self.attiter(x)))
			out.append(self.S[i](x))
		return torch.cat(out, 1)