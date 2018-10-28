import sys
import torch.utils.data
from random import randint
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import cv2
from h5py import File
from scipy.misc import imresize

js=['rank','rkne','rhip','lhip','lknee','lank','pelvi','spin','neck','head','rwr','relb','rshou','lshou','lelb','lwri']
ls=['rloleg','rupleg','rhip','lhip','lupleg','lloleg','rloarm','ruparm','rshou','lshou','luparm','lloarm','spine','head']

edges = [
[1,2], [2,3], [1,0],
[4,0], [4,5], [5,6],
[8,0], [8,9], [9,10],
[11,8], [11,12], [12,13],
[14,8], [14,15], [15,16]
]


def make_heatmap(pts_3d, pts):
	c = np.ones(2) * 256 / 2
	s = 256 * 1.0
	outMap = np.zeros((17, 64, 64))
	outReg = np.zeros((17, 3))
	for i in range(17):
			pt = Transform3D(pts_3d[i], c, s, 0, 64)
			if pts[i][0] > 1:
					outMap[i] = DrawGaussian(outMap[i], pt[:2], 1)
			outReg[i, 2] = pt[2] / 64 * 2 - 1
	return outMap

def test_heatmaps(theatmaps,timg,i):
	"""
	heatmaps=theatmaps[:,:,:]
		heatmaps=heatmaps.transpose(1,2,0)
		# print('heatmap inside shape is',heatmaps.shape)
##    print('----------------here')
##    print(heatmaps.shape)
		img=timg.numpy()
		# print('img inside shape is',img.shape)
		#img=np.squeeze(img)
		# print(img.shape)
		img=img.transpose(1,2,0)
		img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.uint8)
#    print('heatmaps',heatmaps.shape)
		heatmaps = cv2.resize(heatmaps,(0,0), fx=8,fy=8)
#    print('heatmapsafter',heatmaps.shape)
		for j in range(0, 16):
				heatmap = heatmaps[:,:,j]
				heatmap = heatmap.reshape((256,256,1)).astype(np.uint8)
				heatmapimg = np.array(heatmap * 255, dtype = np.uint8)
				heatmap = cv2.applyColorMap(heatmapimg, cv2.COLORMAP_JET)
				heatmap = heatmap/255
				plt.imshow(img)
				plt.imshow(heatmap, alpha=0.5)
				plt.savefig('debug/' + str(i) + '_' + str(j) + '.png')
				# plt.show()
				#plt.savefig('hmtestpadh36'+str(i)+js[j]+'.png')

		print("saved" + str(i))
	"""
	print(theatmaps.shape, timg.shape)
	pass

def draw_heatmaps(heatmaps, image, index):
	img = image
	#print(img.max(), img.min(), img.std(), img.mean())
	img = np.array(255*img.transpose(1, 2, 0), dtype = np.uint16)
	#img = cv2.resize(img, (heatmaps.shape[1], heatmaps.shape[1]))
	#print(img.shape, img.max(), img.min(), img.mean(), img.std())
	#print(img.shape)
	#print(heatmaps.shape[0])
	for i in range(heatmaps.shape[0]):
		#current = cv2.applyColorMap(heatmaps[i, :, :], cv2.COLORMAP_JET)
		current = heatmaps[i, :, :]
		print(current.shape)
		current = cv2.resize(current, (img.shape[0], img.shape[1]))
		#print(current.shape)
		#print(current.mean())
		#print(current.std())
		#print(img.max())
		plt.imshow(img)
		plt.imshow(current, alpha = 0.5)
		plt.savefig('debug/' + str(index) + '_' + str(i) + '.png')
	print("saved", str(index))

