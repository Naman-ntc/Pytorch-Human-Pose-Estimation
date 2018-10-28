import os
import torch
import pickle
from testheatmaps import *
from progress.bar import Bar
from utils import AverageMeter, create_plot_window, adjust_learning_rate

class Trainer(object):
	"""docstring for Trainer"""
	def __init__(self, Model, Optimizer, Loss, Metrics, File, vis, opts):
		super(Trainer, self).__init__()
		self.model = Model
		self.optimizer = Optimizer
		self.Loss = Loss
		self.metrics = Metrics
		self.File = File
		self.opts = opts
		self.gpu = opts.gpuid
		self.model = self.model.to(self.gpu)
		if self.opts.visdom:
			self.vis = vis
			self.epoch_loss_window = create_plot_window(self.vis, '#Epochs', 'Loss', 'Loss Over Epochs')
			self.epoch_pckh_window = create_plot_window(self.vis, '#Epochs', 'PCKh', 'PCKh Over Epochs')

	def test(self, valdataloader):
		with torch.no_grad():
			self._epoch(valdataloader, -1, 'val')

	def train(self, traindataloader, valdataloader, startepoch, endepoch):
		for epoch in range(startepoch, endepoch+1):

			loss_train, pck_train = self._epoch(traindataloader, epoch)


			if epoch%self.opts.valInterval==0:
				with torch.no_grad():
					loss_test, pck_test = self._epoch(valdataloader, epoch, 'val')
				Writer = open(self.File, 'a')
				Writer.write('{:8f} {:4f} {:8f} {:4f} \n'.format(loss_train, pck_train, loss_test, pck_test))
				Writer.close()
			else:
				Writer = open(self.File, 'a')
				Writer.write('{:8f} {:4f} \n'.format(loss_train, pck_train))
				Writer.close()

			if epoch%self.opts.saveInterval==0:
				state = {
					'epoch': epoch+1,
					'model_state': self.model.state_dict(),
					'optimizer_state' : self.optimizer.state_dict(),
				}
				path = os.path.join(self.opts.saveDir, 'model_{}.pth'.format(epoch))
				torch.save(state, path)
			adjust_learning_rate(self.optimizer, epoch, self.opts.dropLR, self.opts.dropMag)
		loss_final = self._epoch(valdataloader, -1, 'val')
		return

	def initepoch(self):
		self.loss = AverageMeter()
		self.loss.reset()
		for key, value in self.metrics.items():
			setattr(self, key, AverageMeter())
		for key, value in self.metrics.items():
			getattr(self, key).reset()
		if self.opts.visdom:
			self.train_iter_loss_window = create_plot_window(self.vis, '#Iterations', 'Loss', 'Training Loss')
			self.train_iter_pck_window = create_plot_window(self.vis, '#Iterations', 'Loss', 'Training PCK')

	def _epoch(self, dataloader, epoch, mode = 'train'):
		"""
		Training logic for an epoch
		"""
		self.initepoch()
		if mode == 'train':
			self.model.train()
		else :
			self.model.eval()

		nIters = len(dataloader)
		bar = Bar('==>', max=nIters)

		for batch_idx, (data, target, meta1, meta2) in enumerate(dataloader):

			data = data.to(self.gpu).float()
			target = target.to(self.gpu).float()
			output = self.model(data)

			loss = self.Loss(output, target)
			self.loss.update(loss.item(), data.shape[0])

			self._eval_metrics(output, target, meta1, meta2, data.shape[0])

			if self.opts.DEBUG:
				draw_heatmaps(target[0,:,:,:].cpu().numpy(), data[0,:,:,:].cpu().numpy(), str(batch_idx) + "tar")
				pass

			if mode == 'train':
				loss.backward()
				if (epoch+1)%self.opts.mini_batch_count==0:
					self.optimizer.step()
					self.optimizer.zero_grad()
					if self.opts.DEBUG:
						pass

			if self.opts.visdom:
				self.plottrain()

			Bar.suffix = mode + ' Epoch: [{0}][{1}/{2}]| Total: {total:} | ETA: {eta:} | Loss: {loss.avg:.6f} ({loss.val:.6f}) | PCKh: {PCKh.avg:.3f} ({PCKh.val:.3f})'.format(epoch, batch_idx+1, nIters, total=bar.elapsed_td, eta=bar.eta_td, loss=self.loss, PCKh=self.PCKh)
			bar.next()
		bar.finish()
		return self.loss.avg, self.PCKh.avg

	def _eval_metrics(self, output, target, meta1, meta2, batchsize):
		for key, value in self.metrics.items():
			value, count = value(output, target, meta1, meta2)
			getattr(self, key).update(value, count)
		return
