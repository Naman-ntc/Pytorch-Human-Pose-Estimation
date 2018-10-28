import torch
import torch.nn as nn
import torch.nn.functional as F


######################################## Chained Predictions
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

				#_stack3.append(nn.ConvTranspose2d(32, 32, 2, 2, 0))
				#_stack3.append(nn.ConvTranspose2d(32, 32, 4, 2, 1))
				#_stack3.append(nn.ConvTranspose2d(32, 32, 6, 2, 2))
				#_stack3.append(nn.BatchNorm2d(32))
				#self.stack3 = nn.ModuleList(_stack3)

				self.end = nn.Conv2d(32, 1, 3, 1, 1)

		def forward(self, x):
				x = self.start(x)
				x = self.stack1[0](x) + self.stack1[1](x) + self.stack1[2](x)
				x = self.stack2[0](x) + self.stack2[1](x) + self.stack2[2](x)
				#x = self.stack3[0](x) + self.stack3[1](x) + self.stack3[2](x)
				x = self.end(x)
				return x



######################################## Stacked HourGlass
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

class myUpsample(nn.Module):
	 def __init__(self):
		 super(myUpsample, self).__init__()
		 pass
	 def forward(self, x):
		 return x[:, :, :, None, :, None].expand(-1, -1, -1, 2, -1, 2).reshape(x.size(0), x.size(1), x.size(2)*2, x.size(3)*2)


class Hourglass(nn.Module):
		"""docstring for Hourglass"""
		def __init__(self, nChannels = 256, numReductions = 4, nModules = 2, poolKernel = (2,2), poolStride = (2,2), upSampleKernel = 2):
				super(Hourglass, self).__init__()
				self.numReductions = numReductions
				self.nModules = nModules
				self.nChannels = nChannels
				self.poolKernel = poolKernel
				self.poolStride = poolStride
				self.upSampleKernel = upSampleKernel
				"""
				For the skip connection, a residual module (or sequence of residuaql modules)
				"""

				_skip = []
				for _ in range(self.nModules):
						_skip.append(Residual(self.nChannels, self.nChannels))

				self.skip = nn.Sequential(*_skip)

				"""
				First pooling to go to smaller dimension then pass input through
				Residual Module or sequence of Modules then  and subsequent cases:
						either pass through Hourglass of numReductions-1
						or pass through Residual Module or sequence of Modules
				"""

				self.mp = nn.MaxPool2d(self.poolKernel, self.poolStride)

				_afterpool = []
				for _ in range(self.nModules):
						_afterpool.append(Residual(self.nChannels, self.nChannels))

				self.afterpool = nn.Sequential(*_afterpool)

				if (numReductions > 1):
						self.hg = Hourglass(self.nChannels, self.numReductions-1, self.nModules, self.poolKernel, self.poolStride)
				else:
						_num1res = []
						for _ in range(self.nModules):
								_num1res.append(Residual(self.nChannels,self.nChannels))

						self.num1res = nn.Sequential(*_num1res)  # doesnt seem that important ?

				"""
				Now another Residual Module or sequence of Residual Modules
				"""

				_lowres = []
				for _ in range(self.nModules):
						_lowres.append(Residual(self.nChannels,self.nChannels))

				self.lowres = nn.Sequential(*_lowres)

				"""
				Upsampling Layer (Can we change this??????)
				As per Newell's paper upsamping recommended
				"""
				self.up = myUpsample()#nn.Upsample(scale_factor = self.upSampleKernel)


		def forward(self, x):
				out1 = x
				out1 = self.skip(out1)
				out2 = x
				out2 = self.mp(out2)
				out2 = self.afterpool(out2)
				if self.numReductions>1:
						out2 = self.hg(out2)
				else:
						out2 = self.num1res(out2)
				out2 = self.lowres(out2)
				out2 = self.up(out2)

				return out2 + out1



######################################## SoftArgMax
import numpy as np
class SoftArgMax(nn.Module):
	"""docstring for SoftArgMax"""
	def __init__(self, opts, factor=None):
		super(SoftArgMax, self).__init__()
		self.softmaxLayer = nn.Softmax(dim=-1)
		self.id = opts.nStack-1
		self.gpuid = opts.gpuid
		self.factor = factor

	def forward(self, x, target):
		x = x[self.id]
		N,C,H,W = x.size()
		reshapedInput = self.factor*x.view(N,C,-1)
		weights = self.softmaxLayer(reshapedInput)
		semiIndices = ((weights) * (torch.arange(H*W).expand(weights.size())).float().to(self.gpuid)).sum(dim=-1)
		indicesY = semiIndices % W
		indicesX = semiIndices / W
		indices = torch.cat((indicesX.unsqueeze(-1), indicesY.unsqueeze(-1)), dim=-1)
		target = torch.from_numpy(self.getPreds(target)).float().to(self.gpuid)
		return F.mse_loss(indices, target)
		#mask = (target > 1e-8).float()
		#print(target < 1e-8, target)
		#return F.mse_loss(indices*mask, target*mask)

	def setFactor(self, factor):
		self.factor = factor

	def getPreds(self, hm):
		hm = hm.cpu().detach().numpy()
		assert len(hm.shape) == 4, 'Input must be a 4-D tensor'
		res = hm.shape[2]
		hm = hm.reshape(hm.shape[0], hm.shape[1], hm.shape[2] * hm.shape[3])
		idx = np.argmax(hm, axis = 2)
		preds = np.zeros((hm.shape[0], hm.shape[1], 2))
		preds[:, :, 0], preds[:, :, 1] = idx % res, idx / res
		return preds