import cv2
import torch
import h5py as  H
import numpy as np
import scipy.io as sio
import datasets.img as I
import torch.utils.data as data
import torchvision.transforms.functional as F

class MPII(data.Dataset):
	def __init__(self, opts, split):
		print('==> initializing 2D {} data.'.format(split))

		self.opts = opts
		self.split = split

		tags = ['imgname','part','center','scale']

		self.stuff1 = sio.loadmat(open(opts.worldCoors[:-4] + ('train' if split is 'train' else '') + '.mat', 'rb'))['a']
		self.stuff2 = sio.loadmat(open(opts.headSize[:-4] + ('train' if split is 'train' else '') + '.mat', 'rb'))['headSize']

		f = H.File('{}/mpii/pureannot/{}.h5'.format(self.opts.dataDir, split), 'r')
		annot = {}
		for tag in tags:
			annot[tag] = np.asarray(f[tag]).copy()
		f.close()
		self.annot = annot

		self.len = len(self.annot['scale'])

		print('Loaded 2D {} {} samples'.format(split, len(annot['scale'])))

	def LoadImage(self, index):
		path = '{}/mpii/images/{}'.format(self.opts.dataDir, ''.join(chr(int(i)) for i in self.annot['imgname'][index]))
		img = cv2.imread(path)
		return img

	def GetPartInfo(self, index):
		pts = self.annot['part'][index].copy()
		c = self.annot['center'][index].copy()
		s = self.annot['scale'][index]
		s = s * 200
		return pts, c, s

	def __getitem__(self, index):
		img = self.LoadImage(index)
		pts, c, s = self.GetPartInfo(index)
		r = 0

		if self.split == 'train':
			s = s * (2 ** I.Rnd(self.opts.maxScale))
			r = 0 if np.random.random() < 0.6 else I.Rnd(self.opts.maxRotate)

		inp = I.Crop(img, c, s, r, self.opts.inputRes) / 256.

		out = np.zeros((self.opts.nJoints, self.opts.outputRes, self.opts.outputRes))

		for i in range(self.opts.nJoints):
			if pts[i][0] > 1:
				pts[i] = I.Transform(pts[i], c, s, r, self.opts.outputRes)
				out[i] = I.DrawGaussian(out[i], pts[i], self.opts.hmGauss, 0.5 if self.opts.outputRes==32 else -1)

		if self.split == 'train':
			if np.random.random() < 0.5:
				inp = I.Flip(inp)
				out = I.ShuffleLR(I.Flip(out))
				pts[:, 0] = self.opts.outputRes/4 - pts[:, 0]

			inp[0] = np.clip(inp[0] * (np.random.random() * (0.4) + 0.6), 0, 1)
			inp[1] = np.clip(inp[1] * (np.random.random() * (0.4) + 0.6), 0, 1)
			inp[2] = np.clip(inp[2] * (np.random.random() * (0.4) + 0.6), 0, 1)

		if self.opts.TargetType=='heatmap':
			return inp, out, self.stuff1[index], self.stuff2[index]
		elif self.opts.TargetType=='direct':
			return inp, np.reshape((pts/self.opts.outputRes), -1) - 0.5, self.stuff1[index], self.stuff2[index]

	def __len__(self):
		return self.len
