# Pytorch-Human-Pose-Estimation
This repository provides implementation with training/testing codes of various human pose estimation architectures in Pytorch
Authors : [Naman Jain](https://github.com/Naman-ntc) and [Sahil Shah](https://github.com/sahil00199)

## Networks Implemented
* [DeepPose](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42237.pdf) - multiple resnet/inception base networks [Pretrained Models Available (MPII and COCO)]
* [Stacked HourGlass Network](https://arxiv.org/pdf/1603.06937.pdf) - standard hoyurglass architecture [Pretrained Models Available (MPII and COCO)]
* [Chained Predictions Network](https://arxiv.org/pdf/1605.02346.pdf) - Sequential prediction of joints [Pretrained Models Available (MPII and COCO)]
* [Pose-Attention](https://arxiv.org/pdf/1702.07432.pdf) - soft attention network
* [PyraNet](https://arxiv.org/pdf/1708.01101.pdf) - pyramid residual modules, fractional maxpooling

### Upcoming Networks
* [IEF](https://arxiv.org/pdf/1507.06550.pdf)
* [DLCM](http://openaccess.thecvf.com/content_ECCV_2018/papers/Wei_Tang_Deeply_Learned_Compositional_ECCV_2018_paper.pdf)

## Datasets
* [MPII](http://human-pose.mpi-inf.mpg.de/)
* [COCO](http://cocodataset.org/#home)

## Requirements
* pytorch == 0.4.1
* torchvision ==0.2.0
* scipy
* configargpare
* progress

## Installation & Setup
`pip install -r requirements.txt`
For setting up MPII dataset please follow [this link](https://github.com/princeton-vl/pose-hg-train#getting-started)
For setting up COCO dataset please follow [this link](https://github.com/Microsoft/human-pose-estimation.pytorch#quick-start)

## Usage
`conf` folder contains the configration files for various different networks along with their options. You can manually tweak different options as per your usage.
Two sets of conf files are required `train/` and `val` for all networks. As it is obvious train files need to be used while training and val files during testing.

`default.defconf` contains generic options for all models such as data-preprocessing, data-path, general optimizer, general training schedule.
Model specific config files contain options for models such as number of layers, base network etc.

To train a model:
	```python main.py -c conf/train/[Model-Name].conf
	-c 	path to the config file containing all options
	```

To validate a model:
	```python main.py -c conf/val/[Model-Name].conf --loadModel [Path-To-Model] -test 
	-c 	path to the config file containing all options
	--loadModel  Path to the saved model
	```

## To Do
* Add links to pretrained models
* Add visualizations and visualization code
* Add code for COCO dataset
* Add more models
I plan (and will try) to complete these very soon!!

