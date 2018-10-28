import os

def create_plot_window(vis, xlabel, ylabel, title):
	return vis.line(X=np.array([1]), Y=np.array([np.nan]), opts=dict(xlabel=xlabel, ylabel=ylabel, title=title))

def adjust_learning_rate(optimizer, epoch, dropLR, dropMag):
	if epoch%dropLR==0:
		lrfac = dropMag
	else:
		lrfac = 1
	for i,param_group in enumerate(optimizer.param_groups):
		if lrfac!=1:
			print("Reducing learning rate of group %d from %f to %f"%(i,param_group['lr'],param_group['lr']*lrfac))
		param_group['lr'] *= lrfac


class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = 0 if self.count is 0 else self.sum / self.count

def ensure_dir(path):
	if path is not None:
		if not os.path.exists(path):
			os.makedirs(path)