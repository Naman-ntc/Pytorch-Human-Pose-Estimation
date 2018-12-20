import coco as c
import torchvision.transforms as transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
dset = c.COCODataset(None, '/home/anurag/datasets/coco', 'train2017', True, transforms.Compose([
            transforms.ToTensor(),
            #normalize,
        ]))

def f(n):
	a = dset.__getitem__(n)
	draw_heatmaps(a[1].numpy(), a[0].numpy(), str(n))

def draw_heatmaps(heatmaps, image, index):
    img = image
    #print(img.max(), img.min(), img.std(), img.mean())
    img = np.array(255*img.transpose(1, 2, 0), dtype = np.uint8)
    #img = cv2.resize(img, (heatmaps.shape[1], heatmaps.shape[1]))
    #print(img.shape, img.max(), img.min(), img.mean(), img.std())
    #print(img.shape)
    #print(heatmaps.shape[0])
    for i in range(heatmaps.shape[0]):
        #current = cv2.applyColorMap(heatmaps[i, :, :], cv2.COLORMAP_JET)
        current = heatmaps[i, :, :]
        current = cv2.resize(current, (img.shape[0], img.shape[1]))
        #print(current.shape)
        #print(current.mean())
        #print(current.std())
        #print(img.max())
        plt.imshow(img)
        plt.imshow(current, alpha = 0.5)
        plt.savefig('debug/' + str(index) + '_' + str(i) + '.png')
    print("saved", str(index))

if __name__ == "__main__":
	print(dset.__len__)
	f(1205)
	f(118000)