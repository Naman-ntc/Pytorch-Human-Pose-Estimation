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

    def PoseAttention(self, output, target, meta1, meta2, alpha=0.5):
        predictions = self.getPreds(output[self.opts.nStack-1].detach().cpu().numpy())
        target = self.getPreds(target.cpu().numpy())
        return self.eval(predictions, target, meta1.cpu().numpy(), meta2.cpu().numpy(), alpha)

    def PyraNet(self, output, target, meta1, meta2, alpha=0.5):
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





class PCK(object):
    """docstring for PCK"""
    def __init__(self, opts):
        super(PCK, self).__init__()
        self.opts = opts
        self.LB = -0.5 + eps if self.opts.TargetType == 'direct' else 0 + eps

    def calc_dists(self, preds, target, normalize):
        preds = preds.astype(np.float32)
        target = target.astype(np.float32)
        dists = np.zeros((preds.shape[1], preds.shape[0]))
        for n in range(preds.shape[0]):
            for c in range(preds.shape[1]):
                if target[n, c, 0] > 1 and target[n, c, 1] > 1:
                    normed_preds = preds[n, c, :] / normalize[n]
                    normed_targets = target[n, c, :] / normalize[n]
                    dists[c, n] = np.linalg.norm(normed_preds - normed_targets)
                else:
                    dists[c, n] = -1
        return dists

    def dist_acc(self, dists, thr=0.5):
         ''' Return percentage below threshold while ignoring values with a -1 '''
         dist_cal = np.not_equal(dists, -1)
         num_dist_cal = dist_cal.sum()
         if num_dist_cal > 0:
             return np.less(dists[dist_cal], thr).sum() * 1.0 / num_dist_cal
         else:
             return -1

    def get_max_preds(self, batch_heatmaps):
        '''
        get predictions from score maps
        heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
        '''
        assert isinstance(batch_heatmaps, np.ndarray), 'batch_heatmaps should be numpy.ndarray'
        assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

        batch_size = batch_heatmaps.shape[0]
        num_joints = batch_heatmaps.shape[1]
        width = batch_heatmaps.shape[3]
        heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
        idx = np.argmax(heatmaps_reshaped, 2)
        maxvals = np.amax(heatmaps_reshaped, 2)

        maxvals = maxvals.reshape((batch_size, num_joints, 1))
        idx = idx.reshape((batch_size, num_joints, 1))

        preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

        preds[:, :, 0] = (preds[:, :, 0]) % width
        preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

        pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
        pred_mask = pred_mask.astype(np.float32)

        preds *= pred_mask
        return preds, maxvals

    def eval(self, pred, target, alpha=0.5):
        '''
        Calculate accuracy according to PCK,
        but uses ground truth heatmap rather than x,y locations
        First value to be returned is average accuracy across 'idxs',
        followed by individual accuracies
        '''
        idx = list(range(16))
        norm = 1.0
        if True:
         h = self.opts.outputRes
         w = self.opts.outputRes
         norm = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10
        dists = self.calc_dists(pred, target, norm)

        acc = np.zeros((len(idx) + 1))
        avg_acc = 0
        cnt = 0

        for i in range(len(idx)):
         acc[i + 1] = self.dist_acc(dists[idx[i]])
         if acc[i + 1] >= 0:
             avg_acc = avg_acc + acc[i + 1]
             cnt += 1

        avg_acc = avg_acc / cnt if cnt != 0 else 0
        if cnt != 0:
         acc[0] = avg_acc
        return avg_acc,cnt

    def StackedHourGlass(self, output, target, meta1, meta2, alpha=0.5):
        predictions = self.get_max_preds(output[self.opts.nStack-1].detach().cpu().numpy())
        target = self.get_max_preds(target.cpu().numpy())
        return self.eval(predictions[0], target[0], alpha)

    def PyraNet(self, output, target, meta1, meta2, alpha=0.5):
        predictions = self.get_max_preds(output[self.opts.nStack-1].detach().cpu().numpy())
        target = self.get_max_preds(target.cpu().numpy())
        return self.eval(predictions[0], target[0], alpha)

    def PoseAttention(self, output, target, meta1, meta2, alpha=0.5):
        predictions = self.get_max_preds(output[self.opts.nStack-1].detach().cpu().numpy())
        target = self.get_max_preds(target.cpu().numpy())
        return self.eval(predictions[0], target[0], alpha)

    def ChainedPredictions(self, output, target, meta1, meta2, alpha=0.5):
        predictions = self.get_max_preds(output.detach().cpu().numpy())
        target = self.get_max_preds(target.cpu().numpy())
        return self.eval(predictions[0], target[0], alpha)

    def DeepPose(self, output, target, meta1, meta2, alpha=0.5):
        predictions = (0. + (output).reshape(-1,16,2).detach().cpu().numpy())*self.opts.outputRes
        target = (0. + (target).reshape(-1,16,2).cpu().numpy())*self.opts.outputRes
        return self.eval(predictions, target, alpha)
