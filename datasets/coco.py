import torch.utils.data as data
from datasets.COCO.coco import COCODataset

class COCO(data.Dataset):
 def __init__(self, opts, split):
	 imageSet = None
	 isTrain = None
	 if split == 'train':
		 imageSet = 'train2017'
		 isTrain = True
	 elif split == 'val':
		 imageSet = 'val2017'
		 isTrain = False
	 import torchvision.transforms as transforms

	 self.stuff = COCODataset(opts, opts.dataDir, imageSet, isTrain, transforms.Compose([
		 transforms.ToTensor(),
	 ]))

 def __getitem__(self, index):
	 return self.stuff.__getitem__(index)

 def __len__(self):
	 return self.stuff.__len__()
