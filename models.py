import torch
import torchvision
import modules as M
import torch.nn as nn
from modules import Hourglass
import torch.nn.functional as F

class StackedHourGlass(nn.Module):
        """docstring for StackedHourGlass"""
        def __init__(self, nChannels, nStack, nModules, numReductions, nJoints):
                super(StackedHourGlass, self).__init__()
                self.nChannels = nChannels
                self.nStack = nStack
                self.nModules = nModules
                self.numReductions = numReductions
                self.nJoints = nJoints

                self.start = M.BnReluConv(3, 64, kernelSize = 7, stride = 2, padding = 3)

                self.res1 = M.Residual(64, 128)
                self.mp = nn.MaxPool2d(2, 2)
                self.res2 = M.Residual(128, 128)
                self.res3 = M.Residual(128, self.nChannels)

                _hourglass, _Residual, _lin1, _chantojoints, _lin2, _jointstochan = [],[],[],[],[],[]

                for _ in range(self.nStack):
                        _hourglass.append(Hourglass(self.nChannels, self.numReductions, self.nModules))
                        _ResidualModules = []
                        for _ in range(self.nModules):
                                _ResidualModules.append(M.Residual(self.nChannels, self.nChannels))
                        _ResidualModules = nn.Sequential(*_ResidualModules)
                        _Residual.append(_ResidualModules)
                        _lin1.append(M.BnReluConv(self.nChannels, self.nChannels))
                        _chantojoints.append(nn.Conv2d(self.nChannels, self.nJoints,1))
                        _lin2.append(nn.Conv2d(self.nChannels, self.nChannels,1))
                        _jointstochan.append(nn.Conv2d(self.nJoints,self.nChannels,1))

                self.hourglass = nn.ModuleList(_hourglass)
                self.Residual = nn.ModuleList(_Residual)
                self.lin1 = nn.ModuleList(_lin1)
                self.chantojoints = nn.ModuleList(_chantojoints)
                self.lin2 = nn.ModuleList(_lin2)
                self.jointstochan = nn.ModuleList(_jointstochan)

        def forward(self, x):
                x = self.start(x)
                x = self.res1(x)
                x = self.mp(x)
                x = self.res2(x)
                x = self.res3(x)
                out = []

                for i in range(self.nStack):
                        x1 = self.hourglass[i](x)
                        x1 = self.Residual[i](x1)
                        x1 = self.lin1[i](x1)
                        out.append(self.chantojoints[i](x1))
                        x1 = self.lin2[i](x1)
                        x = x + x1 + self.jointstochan[i](out[i])

                return (out)

class ChainedPredictions(nn.Module):
        """docstring for ChainedPredictions"""
        def __init__(self, modelName, hhKernel, ohKernel, nJoints):
                super(ChainedPredictions, self).__init__()
                self.nJoints = nJoints
                self.modelName = modelName
                self.resnet = getattr(torchvision.models, self.modelName)(pretrained=True)
                self.resnet.avgpool = M.Identity()
                self.resnet.fc = M.Identity()
                self.hiddenChans = 64 ### Add cases!

                self.hhKernel = hhKernel
                self.ohKernel = ohKernel

                self.init_hidden = nn.Conv2d(512, self.hiddenChans, 1)
                _deception = []
                for i in range(self.nJoints):
                        _deception.append(M.Deception(self.hiddenChans))
                self.deception = nn.ModuleList(_deception)

                _h2h = []
                _o2h = []
                for i in range(nJoints):
                        _o = []
                        _h2h.append(
                                nn.Sequential(
                                        nn.Conv2d(self.hiddenChans, self.hiddenChans, kernel_size=self.hhKernel, padding=self.hhKernel//2),
                                        nn.BatchNorm2d(self.hiddenChans)
                                )
                        )
                        for j in range(i+1):
                                _o.append(nn.Sequential(
                                                nn.Conv2d(1, self.hiddenChans, 1),
                                                nn.Conv2d(self.hiddenChans, self.hiddenChans, kernel_size=self.ohKernel, stride=2, padding=self.ohKernel//2),
                                                nn.BatchNorm2d(self.hiddenChans),
                                                #nn.MaxPool2d(2,2),
                                                #nn.Conv2d(self.hiddenChans, self.hiddenChans, kernel_size=1),
                                                nn.Conv2d(self.hiddenChans, self.hiddenChans, kernel_size=self.ohKernel, stride=2, padding=self.ohKernel//2),
                                                nn.BatchNorm2d(self.hiddenChans),
                                                #nn.MaxPool2d(2,2),
                                                #nn.Conv2d(self.hiddenChans, self.hiddenChans, kernel_size=1)
                                                #nn.Conv2d(self.hiddenChans, self.hiddenChans, kernel_size=self.ohKernel, stride=2, padding=self.ohKernel//2),
                                                #nn.BatchNorm2d(self.hiddenChans),
                                        )
                                )
                        _o2h.append(nn.ModuleList(_o))

                self.h2h = nn.ModuleList(_h2h)
                self.o2h = nn.ModuleList(_o2h)

        def forward(self, x):
                hidden = [0]*self.nJoints
                output = [None]*self.nJoints
                #print(self.resnet(x).shape)
                hidden[0] += self.resnet(x).reshape(-1, 512, 8, 8)
                hidden[0] = self.init_hidden(hidden[0])
                output[0] = self.deception[0](hidden[0])

                for i in range(self.nJoints-1):
                        hidden[i+1] = self.h2h[i](hidden[i])
                        for j in range(i+1):
                                hidden[i+1] += self.o2h[i][j](output[j])
                        hidden[i+1] = torch.relu(hidden[i+1])
                        output[i+1] = self.deception[i+1](hidden[i+1])
                return torch.cat(output, 1)

class DeepPose(nn.Module):
        """docstring for DeepPose"""
        def __init__(self, nJoints, modelName='resnet50'):
                super(DeepPose, self).__init__()
                self.nJoints = nJoints
                self.block = 'BottleNeck' if (int(modelName[6:]) > 34) else 'BasicBlock'
                self.resnet = getattr(torchvision.models, modelName)(pretrained=True)
                self.resnet.fc = nn.Sequential(
                        #nn.BatchNorm1d(512*4 * (4 if self.block == 'BottleNeck' else 1)),
                        nn.Linear(512*4 * (4 if self.block == 'BottleNeck' else 1), self.nJoints * 2)
                )
        def forward(self, x):
                return self.resnet(x)
