import os
import torch
import pickle
from progress.bar import Bar
from utils import AverageMeter, adjust_learning_rate

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
                self.model = self.model

        def test(self, valdataloader):
                with torch.no_grad():
                        self._epoch(valdataloader, -1, 'val')

        def train(self, traindataloader, valdataloader, startepoch, endepoch):
                for epoch in range(startepoch, endepoch+1):

                        train = self._epoch(traindataloader, epoch)

                        if epoch%self.opts.valInterval==0:
                                with torch.no_grad():
                                        test = self._epoch(valdataloader, epoch, 'val')
                                Writer = open(self.File, 'a')
                                Writer.write(train + ' ' + test + '\n')
                                Writer.close()
                        else:
                                Writer = open(self.File, 'a')
                                Writer.write(train + '\n')
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
                        model = self.model.to(self.gpu)
                        data = data.to(self.gpu, non_blocking=True).float()
                        target = target.to(self.gpu, non_blocking=True).float()
                        output = model(data)

                        loss = self.Loss(output, target, meta1.to(self.gpu, non_blocking=True).float().unsqueeze(-1))
                        self.loss.update(loss.item(), data.shape[0])

                        self._eval_metrics(output, target, meta1, meta2, data.shape[0])

                        if self.opts.DEBUG:
                                pass

                        if mode == 'train':
                                loss.backward()
                                if (epoch+1)%self.opts.mini_batch_count==0:
                                        self.optimizer.step()
                                        self.optimizer.zero_grad()
                                        if self.opts.DEBUG:
                                                pass

                        Bar.suffix = mode + ' Epoch: [{0}][{1}/{2}]| Total: {total:} | ETA: {eta:} | Loss: {loss.avg:.6f} ({loss.val:.6f})'.format(epoch, batch_idx+1, nIters, total=bar.elapsed_td, eta=bar.eta_td, loss=self.loss) + self._print_metrics()
                        bar.next()
                bar.finish()
                return '{:8f} '.format(self.loss.avg) + ' '.join(['{:4f}'.format(getattr(self, key).avg) for key,_ in self.metrics.items()])

        def _eval_metrics(self, output, target, meta1, meta2, batchsize):
                for key, value in self.metrics.items():
                        value, count = value(output, target, meta1, meta2)
                        getattr(self, key).update(value, count)
                return

        def _print_metrics(self):
                return ''.join([('| {0}: {metric.avg:.3f} ({metric.val:.3f}) '.format(key, metric=getattr(self, key))) for key, _ in self.metrics.items()])
