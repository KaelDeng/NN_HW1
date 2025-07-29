import numpy as np
import os
import time
import matplotlib.pyplot as plt

class SgdOpt:
    """
    SGD optimizer class.
    """
    def __init__(self, lr=0.01):
        self.lr = lr
    def update(self, paramDict, gradDict):
        for key in paramDict.keys():
            paramDict[key] -= self.lr * gradDict[key]

class LrScheduler:
    """
    Learning rate scheduler class.
    """
    @staticmethod
    def stepDecay(initLr, epoch, drop=0.5, dropEpoch=10):
        factor = np.power(drop, np.floor((1 + epoch) / dropEpoch))
        return initLr * factor
    @staticmethod
    def expDecay(initLr, epoch, decay=0.95):
        return initLr * (decay ** epoch)

class NetTrainer:
    """
    Neural network trainer class.
    """
    def __init__(self, net, opt, lrSched=None, savePath='./cifar10/results'):
        self.net = net
        self.opt = opt
        self.lrSched = lrSched
        self.savePath = savePath
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        self.lossTrainList = []
        self.lossValList = []
        self.accTrainList = []
        self.accValList = []
        self.bestValAcc = 0
        self.initLr = opt.lr
    def train(self, Xtr, Ytr, Xval, Yval, nEpoch=50, batchSz=100, regBeta=0.0, verbose=True, recordStep=1):
        nTrain = Xtr.shape[0]
        nIter = max(nTrain // batchSz, 1)
        maxIter = nEpoch * nIter
        tStart = time.time()
        for ep in range(1, nEpoch + 1):
            if self.lrSched:
                self.opt.lr = self.lrSched(self.initLr, ep)
            epStart = time.time()
            idx = np.arange(nTrain)
            np.random.shuffle(idx)
            XtrShuf = Xtr[idx]
            YtrShuf = Ytr[idx]
            epLoss = 0
            for i in range(nIter):
                bStart = i * batchSz
                bEnd = min(bStart + batchSz, nTrain)
                Xb = XtrShuf[bStart:bEnd]
                Yb = YtrShuf[bStart:bEnd]
                lossVal = self._step(Xb, Yb, regBeta)
                epLoss += lossVal
            epLoss /= nIter
            if ep % recordStep == 0:
                trLoss = self.net.loss(Xtr, Ytr, regBeta)
                trAcc = self.net.accuracy(Xtr, Ytr)
                self.lossTrainList.append(trLoss)
                self.accTrainList.append(trAcc)
                valLoss = self.net.loss(Xval, Yval, regBeta)
                valAcc = self.net.accuracy(Xval, Yval)
                self.lossValList.append(valLoss)
                self.accValList.append(valAcc)
                if valAcc > self.bestValAcc:
                    self.bestValAcc = valAcc
                    self.net.saveParams(os.path.join(self.savePath, 'best_model.pkl'))
                epTime = time.time() - epStart
                if verbose:
                    print(f"Epoch {ep}/{nEpoch}, time cost: {epTime:.2f}s, lr: {self.opt.lr:.6f}")
                    print(f"Traning loss: {trLoss:.4f}, Traning ACC: {trAcc:.4f}")
                    print(f"Validation loss: {valLoss:.4f}, Validation ACC: {valAcc:.4f}")
                    print("-" * 50)
        self.net.saveParams(os.path.join(self.savePath, 'final_model.pkl'))
        tTotal = time.time() - tStart
        if verbose:
            print(f"Training finished, total time: {tTotal:.2f}s")
            print(f"Best Training ACC: {self.bestValAcc:.4f}")
    def _step(self, Xb, Yb, regBeta):
        self.net.forward(Xb)
        lossVal = self.net.lossObj.forward(self.net.layer3Z, Yb)
        if regBeta > 0:
            W1, W2, W3 = self.net.paramDict['W1'], self.net.paramDict['W2'], self.net.paramDict['W3']
            regLoss = 0.5 * regBeta * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2))
            lossVal += regLoss
        gradDict = self.net.backward(Yb, regBeta)
        self.opt.update(self.net.paramDict, gradDict)
        return lossVal
    def plotCurve(self, savePath=None):
        if savePath is None:
            savePath = self.savePath
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        epIdx = range(1, len(self.lossTrainList) + 1)
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(epIdx, self.lossTrainList, label='Train Loss')
        plt.plot(epIdx, self.lossValList, label='Val Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(epIdx, self.accTrainList, label='Train Acc')
        plt.plot(epIdx, self.accValList, label='Val Acc')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Curve')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(savePath, 'train_curve.png'))
        plt.close()
    def showWeights(self, savePath=None, whichLayer=1, reshape=True):
        if savePath is None:
            savePath = self.savePath
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        if whichLayer == 1:
            weights = self.net.paramDict['W1']
            title = 'Layer1 Weights'
        elif whichLayer == 2:
            weights = self.net.paramDict['W2']
            title = 'Layer2 Weights'
        elif whichLayer == 3:
            weights = self.net.paramDict['W3']
            title = 'Layer3 Weights'
        else:
            raise ValueError('Layer index must be 1, 2 or 3')
        if whichLayer == 1 and reshape:
            nShow = min(100, weights.shape[1])
            gridSz = int(np.ceil(np.sqrt(nShow)))
            plt.figure(figsize=(15, 15))
            for i in range(nShow):
                if i >= gridSz * gridSz:
                    break
                plt.subplot(gridSz, gridSz, i+1)
                img = weights[:, i].reshape(32, 32, 3)
                img = (img - img.min()) / (img.max() - img.min())
                plt.imshow(img)
                plt.axis('off')
            plt.suptitle(f'{title} (as Images)', fontsize=16)
            plt.tight_layout()
            plt.subplots_adjust(top=0.95)
            plt.savefig(os.path.join(savePath, f'layer{whichLayer}_img.png'))
            plt.close()
        plt.figure(figsize=(12, 10))
        if weights.size > 1000000:
            maxElem = 1000
            rows = min(weights.shape[0], int(np.sqrt(maxElem)))
            cols = min(weights.shape[1], int(np.sqrt(maxElem)))
            rowIdx = np.random.choice(weights.shape[0], rows, replace=False)
            colIdx = np.random.choice(weights.shape[1], cols, replace=False)
            sampled = weights[np.ix_(rowIdx, colIdx)]
            plt.title(f'{title} (Sample {rows}x{cols})')
            plt.imshow(sampled, cmap='viridis')
        else:
            plt.title(title)
            plt.imshow(weights, cmap='viridis')
        plt.colorbar()
        plt.savefig(os.path.join(savePath, f'layer{whichLayer}_heatmap.png'))
        plt.close() 