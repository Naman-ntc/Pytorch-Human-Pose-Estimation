import datasets
from torch.utils.data import DataLoader

def ImageLoader(opts, split):
	return DataLoader(
			dataset = getattr(datasets, opts.dataset)(opts, split),
			batch_size = opts.data_loader_size,
			shuffle = opts.shuffle if split=='train' else False,
			pin_memory = not(opts.dont_pin_memory),
			num_workers = opts.nThreads
	)
