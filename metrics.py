import torch
import numpy as np
eps = 1e-8

class PCKh(object):
    """docstring for PCKh"""
    def __init__(self, opts):
        super(PCKh, self).__init__()
        self.opts = opts
        self.LB = -0.5 + eps if self.opts.TargetType == 'direct' else 0 + eps

    def getPreds(self, hm):
        assert len(hm.shape) == 4, 'Input must be a 4-D tensor'
        res = hm.shape[2]
        hm = hm.reshape(hm.shape[0], hm.shape[1], hm.shape[2] * hm.shape[3])
        idx = np.argmax(hm, axis = 2)
        preds = np.zeros((hm.shape[0], hm.shape[1], 2))
        preds[:, :, 0], preds[:, :, 1] = idx % res, idx / res
        return preds

    def eval(self, predictions, target, meta1, meta2, alpha=0.5):
        batchSize = predictions.shape[0]
        numJoints = 0
        numCorrect = 0
        for i in range(batchSize):
            index1 = 0
            index2 = 0
            skip = 0
            while (np.isnan(meta1[i,index1,:]).any() or (target[i,index1,:]<= self.LB).any()):
                index1+=1
                if index1>=15:
                    skip = 1
                    break
            if skip:
                continue
            index2 = index1 + 1
            while (np.isnan(meta1[i,index2,:]).any() or (meta1[i,index2, :]==meta1[i,index1,:]).all() or (target[i,index2,:]<= self.LB).any() or (target[i,index2, :]==target[i,index1,:]).all()):
                index2+=1
                if index2>=16:
                    skip = 1
                    break
            if skip:
                continue

            # Found 2 non-nan indices

            loaderDist = np.linalg.norm(target[i, index1, :] - target[i, index2, :])
            globalDist = np.linalg.norm(meta1[i, index1, :] - meta1[i, index2, :])
            effectiveHeadSize = meta2[i, 0] * (loaderDist/globalDist)

            for j in range(16):
                    if j==7 or j==6:
                        continue
                    if target[i, j, 0] >= self.LB and target[i, j, 1] >= self.LB and not(np.isnan(meta1[i, j, :]).any()):
                            numJoints += 1
                            if np.linalg.norm(predictions[i, j, :] - target[i, j, :]) <= alpha * effectiveHeadSize:
                                    numCorrect += 1
        if numJoints == 0:
            return 1, 0
        return float(numCorrect)/float(numJoints), numJoints

    def StackedHourGlass(self, output, target, meta1, meta2, alpha=0.5):
        predictions = self.getPreds(output[self.opts.nStack-1].detach().cpu().numpy())
        target = self.getPreds(target.cpu().numpy())
        return self.eval(predictions, target, meta1.cpu().numpy(), meta2.cpu().numpy(), alpha)

    def ChainedPredictions(self, output, target, meta1, meta2, alpha=0.5):
        predictions = self.getPreds(output.detach().cpu().numpy())
        target = self.getPreds(target.cpu().numpy())
        return self.eval(predictions, target, meta1.cpu().numpy(), meta2.cpu().numpy(), alpha)

    def DeepPose(self, output, target, meta1, meta2, alpha=0.5):
        predictions = (output).reshape(-1,16,2).detach().cpu().numpy()
        target = (target).reshape(-1,16,2).cpu().numpy()
        return self.eval(predictions, target, meta1.cpu().numpy(), meta2.cpu().numpy(), alpha)
