import os
import ref
import datetime
import utils as U
import configargparse

class opts():
        """docstring for opts"""
        def __init__(self):
                super(opts, self).__init__()
                self.parser = configargparse.ArgParser(default_config_files=['default.defconf'])

        def init(self):
                self.parser.add('-test', action='store_true', help='test')
                self.parser.add('-usedouble', action='store_true', help='Default tensor type to float')
                self.parser.add('-DEBUG', action='store_true', help='To run in debug mode')
                self.parser.add('-dump', action='store_true', help='To run in debug mode')
                self.parser.add('-dont_pin_memory', default=0, help='Whether to pin memory or not')

                self.parser.add('-c', '--my-config', required=False, is_config_file=True, help='config file path')

                self.parser.add('--expDir', help='Experiment-ID')
                self.parser.add('--expID', help='Experiment-ID')

                self.parser.add('--visdom', type=int, help='Batch Size for DataLoader')

                self.parser.add('--model', help='Which model to use')

                self.parser.add('--loadModel', help='Path to the model to load')
                self.parser.add('--loadOptim', type=int, help='Path to the model to load')
                self.parser.add('--dropPreLoaded', type=int, help='Path to the model to load')
                self.parser.add('--dropMagPreLoaded', type=float, help='Path to the model to load')
                self.parser.add('--loadEpoch', type=int, help='Path to the model to load')

                self.parser.add('--translate', type=float, help='Maximum translation as a percentage of the image width')
                self.parser.add('--maxScale', type=float, help='How much to scale the image or zoom in')
                self.parser.add('--maxRotate', type=float, help='maximum angle of rotation on either side')

                self.parser.add('--dataDir', help='Directory for the data')
                self.parser.add('--scale', type=float, help= 'Batch Size for Dataloader')
                self.parser.add('--rotate', type=int, help='Batch Size for DataLoader')
                self.parser.add('--inputRes', type=int, help='Batch Size for DataLoader')
                self.parser.add('--imageRes', type=int, help='Batch Size for DataLoader')
                self.parser.add('--outputRes', type=int, help='Batch Size for DataLoader')
                self.parser.add('--inputDim', type=int, help='this is originally 224')
                self.parser.add('--hmGauss', type=int, help='Batch Size for DataLoader')
                self.parser.add('--hmGaussInp', type=int, help='Batch Size for DataLoader')

                self.parser.add('--TargetType', help='Batch Size for DataLoader')


                self.parser.add('--modelName', help='Batch Size for DataLoader')
                self.parser.add('--hhKernel', type=int, help='Batch Size for DataLoader')
                self.parser.add('--ohKernel', type=int, help='Batch Size for DataLoader')

                self.parser.add('--baseName', help='Batch Size for DataLoader')

                self.parser.add('--nChannels', type=int, help='Batch Size for DataLoader')
                self.parser.add('--nStack', type=int, help='Batch Size for DataLoader')
                self.parser.add('--nModules', type=int, help='Batch Size for DataLoader')
                self.parser.add('--nReductions', type=int, help='Batch Size for DataLoader')


                self.parser.add('--nJoints', type=int, help='Batch Size for DataLoader')

                self.parser.add('--dataset', help='Shuffle the data during training')
                self.parser.add('--shuffle', type=int, help='Shuffle the data during training')
                self.parser.add('--augment', type=int, help='Perform Augmentation during data loading')
                self.parser.add('--valInterval', type=int, help='After how many train epoch to run a val epoch')
                self.parser.add('--saveInterval', type=int, help='After how many train epochs to save model')
                self.parser.add('--nThreads', type=int, help='How many threads to use for Dataloader')

                self.parser.add('--gpuid', type=int, help='GPU ID for the model')
                self.parser.add('--nEpochs', type=int, help='Number of epochs to train')

                self.parser.add('--data_loader_size', type=int, help='Batch Size for DataLoader')
                self.parser.add('--mini_batch_count', type=int, help='After how many mini batches to run backprop')
                self.parser.add('--optimizer_type', help='Which optimizer to use in DataLoader')
                self.parser.add('--optimizer_pars', action='append' , help='parameters for the optimizer')
                self.parser.add('--LR', type=float, help='Learning rate for the base resnet')

                self.parser.add('--dropLR', type=int, help='Drop LR after how many epochs')
                self.parser.add('--dropMag', type=float, help='Drop LR after how many epochs')

                self.parser.add('--worldCoors', help='World Coordinates file path')
                self.parser.add('--headSize', help='head Size file path')

        def parse(self):
                self.init()
                self.opt = self.parser.parse_args()
                if self.opt.DEBUG:
                        self.opt.data_loader_size = 1
                        self.opt.shuffle = 0

                self.opt.saveDir = os.path.join(os.path.join(self.opt.expDir, self.opt.expID), os.path.join(self.opt.model, 'logs_{}'.format(datetime.datetime.now().isoformat())))
                self.opt.saveDir = os.path.join(self.opt.expDir, self.opt.model, self.opt.expID, 'logs_{}'.format(datetime.datetime.now().isoformat()))
                U.ensure_dir(self.opt.saveDir)

                ####### Write All Opts
                args = dict((name, getattr(self.opt, name)) for name in dir(self.opt)
                                        if not name.startswith('_'))
                refs = dict((name, getattr(ref, name)) for name in dir(ref)
                                        if not name.startswith('_'))

                file_name = os.path.join(self.opt.saveDir, 'opt.txt')
                with open(file_name, 'wt') as opt_file:
                        opt_file.write('==> Args:\n')
                        for k, v in sorted(args.items()):
                                opt_file.write("%s: %s\n"%(str(k), str(v)))

                return self.opt
