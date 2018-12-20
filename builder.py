import torch
import models
import losses
import metrics
import dataloaders as dataloaders


class Builder(object):
	"""docstring for Builder"""
	def __init__(self, opts):
		super(Builder, self).__init__()
		self.opts = opts
		if opts.loadModel is not None:
			self.states = torch.load(opts.loadModel)
		else:
			self.states = None
	def Model(self):
		ModelBuilder = getattr(models, self.opts.model)
		if self.opts.model == 'StackedHourGlass':
			Model = ModelBuilder(self.opts.nChannels, self.opts.nStack, self.opts.nModules, self.opts.nReductions, self.opts.nJoints)
		elif self.opts.model == 'ChainedPredictions':
			Model = ModelBuilder(self.opts.modelName, self.opts.hhKernel, self.opts.ohKernel, self.opts.nJoints)
		elif self.opts.model == 'DeepPose':
			Model = ModelBuilder(self.opts.nJoints, self.opts.baseName)
		elif self.opts.model == 'PyraNet':
			Model = ModelBuilder(self.opts.nChannels, self.opts.nStack, self.opts.nModules, self.opts.nReductions, self.opts.baseWidth, self.opts.cardinality, self.opts.nJoints, self.opts.inputRes)
		elif self.opts.model == 'PoseAttention':
			Model = ModelBuilder(self.opts.nChannels, self.opts.nStack, self.opts.nModules, self.opts.nReductions, self.opts.nJoints, self.LRNSize, self.opts.IterSize)
		else:
			assert('Not Implemented Yet!!!')
		if self.states is not None:
			Model.load_state_dict(self.states['model_state'])
		return Model

	def Loss(self):
		instance = losses.Loss(self.opts)
		return getattr(instance, self.opts.model)

	def Metric(self):
		PCKhinstance = metrics.PCKh(self.opts)
		PCKinstance = metrics.PCK(self.opts)
		if self.opts.dataset=='MPII':
			return {'PCK' : getattr(PCKinstance, self.opts.model), 'PCKh' : getattr(PCKhinstance, self.opts.model)}         
		if self.opts.dataset=='COCO':
			return {'PCK' : getattr(PCKinstance, self.opts.model)}
			
	def Optimizer(self, Model):
		TrainableParams = filter(lambda p: p.requires_grad, Model.parameters())
		Optimizer = getattr(torch.optim, self.opts.optimizer_type)(TrainableParams, lr = self.opts.LR, alpha = 0.99, eps = 1e-8)
		if self.states is not None and self.opts.loadOptim:
			Optimizer.load_state_dict(states['optimizer_state'])
			if self.opts.dropPreLoaded:
				for i,_ in enumarate(Optimizer.param_groups):
					Optimizer.param_groups[i]['lr'] /= opts.dropMagPreLoaded
		return Optimizer

	def DataLoaders(self):
		return dataloaders.ImageLoader(self.opts, 'train'), dataloaders.ImageLoader(self.opts, 'val')

	def Epoch(self):
		Epoch = 1
		if self.states is not None and self.opts.loadEpoch:
			Epoch = states['epoch']
		return Epoch
