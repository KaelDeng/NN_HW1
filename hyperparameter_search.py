import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import time
from model import ThreeLayerClassifier
from trainer import SgdOpt, LrScheduler, NetTrainer
from data import load_cifar10, DataSplitter

class ParamSearcher:
    """
    Hyperparameter searcher for CIFAR-10 dataset.
    """
    def __init__(self, savePath='./cifar10/hyperparam_results'):
        self.savePath = savePath
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        self.resultList = []
    def searchLr(self, Xtr, Ytr, Xval, Yval, hid1=100, hid2=100, lrList=None, actType='relu', batchSz=100, nEpoch=20, regBeta=0.0, recordStep=5, verbose=True):
        if lrList is None:
            lrList = [0.001, 0.01, 0.05, 0.1, 0.5]
        bestValAcc = 0
        bestLr = None
        lrRes = []
        for lr in lrList:
            if verbose:
                print(f"\nTesting learning rate: {lr}")
            net = ThreeLayerClassifier(
                inDim=Xtr.shape[1],
                hidDim1=hid1,
                hidDim2=hid2,
                outDim=10,
                actType=actType)
            opt = SgdOpt(lr=lr)
            saveDir = os.path.join(self.savePath, f'lr_{lr}')
            trainer = NetTrainer(net, opt, savePath=saveDir)
            trainer.train(Xtr, Ytr, Xval, Yval, nEpoch=nEpoch, batchSz=batchSz, regBeta=regBeta, verbose=verbose, recordStep=recordStep)
            res = {
                'lr': lr,
                'hid1': hid1,
                'hid2': hid2,
                'actType': actType,
                'regBeta': regBeta,
                'valAcc': trainer.bestValAcc,
                'trainLoss': trainer.lossTrainList,
                'valLoss': trainer.lossValList,
                'trainAcc': trainer.accTrainList,
                'valAccList': trainer.accValList
            }
            lrRes.append(res)
            self.resultList.append(res)
            if trainer.bestValAcc > bestValAcc:
                bestValAcc = trainer.bestValAcc
                bestLr = lr
            trainer.plotCurve()
        self._plotLrCompare(lrRes)
        if verbose:
            print(f"\nLearning rate search finished! Best LR: {bestLr}, Best val ACC: {bestValAcc:.4f}")
        return bestLr
    def searchHidden(self, Xtr, Ytr, Xval, Yval, hiddenList=None, lr=0.01, actType='relu', batchSz=100, nEpoch=20, regBeta=0.0, recordStep=5, verbose=True):
        if hiddenList is None:
            hiddenList = [(50, 50), (100, 100), (200, 100), (300, 150), (500, 200)]
        bestValAcc = 0
        bestHidden = None
        hiddenRes = []
        for h1, h2 in hiddenList:
            if verbose:
                print(f"\nTesting hidden size ({h1}, {h2})")
            net = ThreeLayerClassifier(
                inDim=Xtr.shape[1],
                hidDim1=h1,
                hidDim2=h2,
                outDim=10,
                actType=actType)
            opt = SgdOpt(lr=lr)
            saveDir = os.path.join(self.savePath, f'hidden_{h1}_{h2}')
            trainer = NetTrainer(net, opt, savePath=saveDir)
            trainer.train(Xtr, Ytr, Xval, Yval, nEpoch=nEpoch, batchSz=batchSz, regBeta=regBeta, verbose=verbose, recordStep=recordStep)
            res = {
                'lr': lr,
                'hid1': h1,
                'hid2': h2,
                'actType': actType,
                'regBeta': regBeta,
                'valAcc': trainer.bestValAcc,
                'trainLoss': trainer.lossTrainList,
                'valLoss': trainer.lossValList,
                'trainAcc': trainer.accTrainList,
                'valAccList': trainer.accValList
            }
            hiddenRes.append(res)
            self.resultList.append(res)
            if trainer.bestValAcc > bestValAcc:
                bestValAcc = trainer.bestValAcc
                bestHidden = (h1, h2)
            trainer.plotCurve()
        self._plotHiddenCompare(hiddenRes)
        if verbose:
            print(f"\nHidden size search finished! Best: {bestHidden}, Best val ACC: {bestValAcc:.4f}")
        return bestHidden
    def searchReg(self, Xtr, Ytr, Xval, Yval, hid1=100, hid2=100, regList=None, lr=0.01, actType='relu', batchSz=100, nEpoch=20, recordStep=5, verbose=True):
        if regList is None:
            regList = [0.0, 0.001, 0.01, 0.1, 0.5]
        bestValAcc = 0
        bestReg = None
        regRes = []
        for reg in regList:
            if verbose:
                print(f"\nTesting regularization {reg}")
            net = ThreeLayerClassifier(
                inDim=Xtr.shape[1],
                hidDim1=hid1,
                hidDim2=hid2,
                outDim=10,
                actType=actType)
            opt = SgdOpt(lr=lr)
            saveDir = os.path.join(self.savePath, f'reg_{reg}')
            trainer = NetTrainer(net, opt, savePath=saveDir)
            trainer.train(Xtr, Ytr, Xval, Yval, nEpoch=nEpoch, batchSz=batchSz, regBeta=reg, verbose=verbose, recordStep=recordStep)
            res = {
                'lr': lr,
                'hid1': hid1,
                'hid2': hid2,
                'actType': actType,
                'regBeta': reg,
                'valAcc': trainer.bestValAcc,
                'trainLoss': trainer.lossTrainList,
                'valLoss': trainer.lossValList,
                'trainAcc': trainer.accTrainList,
                'valAccList': trainer.accValList
            }
            regRes.append(res)
            self.resultList.append(res)
            if trainer.bestValAcc > bestValAcc:
                bestValAcc = trainer.bestValAcc
                bestReg = reg
            trainer.plotCurve()
        self._plotRegCompare(regRes)
        if verbose:
            print(f"\nRegularization search finished! Best: {bestReg}, Best val ACC: {bestValAcc:.4f}")
        return bestReg
    def searchAct(self, Xtr, Ytr, Xval, Yval, hid1=100, hid2=100, actList=None, lr=0.01, batchSz=100, nEpoch=20, regBeta=0.0, recordStep=5, verbose=True):
        if actList is None:
            actList = ['relu', 'sigmoid', 'tanh']
        bestValAcc = 0
        bestAct = None
        actRes = []
        for actType in actList:
            if verbose:
                print(f"\nTesting activation function {actType}")
            net = ThreeLayerClassifier(
                inDim=Xtr.shape[1],
                hidDim1=hid1,
                hidDim2=hid2,
                outDim=10,
                actType=actType)
            opt = SgdOpt(lr=lr)
            saveDir = os.path.join(self.savePath, f'act_{actType}')
            trainer = NetTrainer(net, opt, savePath=saveDir)
            trainer.train(Xtr, Ytr, Xval, Yval, nEpoch=nEpoch, batchSz=batchSz, regBeta=regBeta, verbose=verbose, recordStep=recordStep)
            res = {
                'lr': lr,
                'hid1': hid1,
                'hid2': hid2,
                'actType': actType,
                'regBeta': regBeta,
                'valAcc': trainer.bestValAcc,
                'trainLoss': trainer.lossTrainList,
                'valLoss': trainer.lossValList,
                'trainAcc': trainer.accTrainList,
                'valAccList': trainer.accValList
            }
            actRes.append(res)
            self.resultList.append(res)
            if trainer.bestValAcc > bestValAcc:
                bestValAcc = trainer.bestValAcc
                bestAct = actType
            trainer.plotCurve()
        self._plotActCompare(actRes)
        if verbose:
            print(f"\nActivation function search finished! Best: {bestAct}, Best val ACC: {bestValAcc:.4f}")
        return bestAct
    def _plotLrCompare(self, resList):
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        lrList = [r['lr'] for r in resList]
        valAccList = [r['valAcc'] for r in resList]
        plt.bar(range(len(lrList)), valAccList)
        plt.xticks(range(len(lrList)), [str(lr) for lr in lrList])
        plt.xlabel('Learning Rate')
        plt.ylabel('Val Acc')
        plt.title('Val Acc vs LR')
        plt.subplot(1, 2, 2)
        ep = range(1, len(resList[0]['valLoss']) + 1)
        for r in resList:
            plt.plot(ep, r['valLoss'], label=f"lr={r['lr']}")
        plt.xlabel('Epochs')
        plt.ylabel('Val Loss')
        plt.title('Val Loss vs LR')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.savePath, 'lr_compare.png'))
        plt.close()
    def _plotHiddenCompare(self, resList):
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        hiddenList = [f"({r['hid1']}, {r['hid2']})" for r in resList]
        valAccList = [r['valAcc'] for r in resList]
        plt.bar(range(len(hiddenList)), valAccList)
        plt.xticks(range(len(hiddenList)), hiddenList, rotation=45)
        plt.xlabel('Hidden Size (1,2)')
        plt.ylabel('Val Acc')
        plt.title('Val Acc vs Hidden Size')
        plt.subplot(1, 2, 2)
        ep = range(1, len(resList[0]['valLoss']) + 1)
        for r in resList:
            plt.plot(ep, r['valLoss'], label=f"hidden=({r['hid1']},{r['hid2']})")
        plt.xlabel('Epochs')
        plt.ylabel('Val Loss')
        plt.title('Val Loss vs Hidden Size')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.savePath, 'hidden_compare.png'))
        plt.close()
    def _plotRegCompare(self, resList):
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        regList = [r['regBeta'] for r in resList]
        valAccList = [r['valAcc'] for r in resList]
        plt.bar(range(len(regList)), valAccList)
        plt.xticks(range(len(regList)), [str(r) for r in regList])
        plt.xlabel('Reg Beta')
        plt.ylabel('Val Acc')
        plt.title('Val Acc vs Reg')
        plt.subplot(1, 2, 2)
        ep = range(1, len(resList[0]['valLoss']) + 1)
        for r in resList:
            plt.plot(ep, r['valLoss'], label=f"reg={r['regBeta']}")
        plt.xlabel('Epochs')
        plt.ylabel('Val Loss')
        plt.title('Val Loss vs Reg')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.savePath, 'reg_compare.png'))
        plt.close()
    def _plotActCompare(self, resList):
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        actList = [r['actType'] for r in resList]
        valAccList = [r['valAcc'] for r in resList]
        plt.bar(range(len(actList)), valAccList)
        plt.xticks(range(len(actList)), actList)
        plt.xlabel('Act Func')
        plt.ylabel('Val Acc')
        plt.title('Val Acc vs Act')
        plt.subplot(1, 2, 2)
        ep = range(1, len(resList[0]['valLoss']) + 1)
        for r in resList:
            plt.plot(ep, r['valLoss'], label=f"{r['actType']}")
        plt.xlabel('Epochs')
        plt.ylabel('Val Loss')
        plt.title('Val Loss vs Act')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.savePath, 'act_compare.png'))
        plt.close()
    def saveRes(self, fileName='search_res.pkl'):
        with open(os.path.join(self.savePath, fileName), 'wb') as f:
            pickle.dump(self.resultList, f)
    def loadRes(self, fileName='search_res.pkl'):
        with open(os.path.join(self.savePath, fileName), 'rb') as f:
            self.resultList = pickle.load(f)
    def gridSearch(self, Xtr, Ytr, Xval, Yval, lrList=None, hiddenList=None, regList=None, actList=None, batchSz=100, nEpoch=10, recordStep=5, verbose=True):
        if lrList is None:
            lrList = [0.01, 0.1]
        if hiddenList is None:
            hiddenList = [(100, 100), (200, 100)]
        if regList is None:
            regList = [0.0, 0.01]
        if actList is None:
            actList = ['relu', 'tanh']
        bestValAcc = 0
        bestParams = None
        tStart = time.time()
        totalComb = len(lrList) * len(hiddenList) * len(regList) * len(actList)
        curComb = 0
        for lr in lrList:
            for hidden in hiddenList:
                h1, h2 = hidden
                for reg in regList:
                    for actType in actList:
                        curComb += 1
                        if verbose:
                            print(f"\n[{curComb}/{totalComb}] Testing: lr={lr}, hidden=({h1},{h2}), reg={reg}, act={actType}")
                        net = ThreeLayerClassifier(
                            inDim=Xtr.shape[1],
                            hidDim1=h1,
                            hidDim2=h2,
                            outDim=10,
                            actType=actType)
                        opt = SgdOpt(lr=lr)
                        saveDir = os.path.join(self.savePath, f'grid_lr{lr}_h{h1}_{h2}_reg{reg}_act{actType}')
                        trainer = NetTrainer(net, opt, savePath=saveDir)
                        trainer.train(Xtr, Ytr, Xval, Yval, nEpoch=nEpoch, batchSz=batchSz, regBeta=reg, verbose=verbose, recordStep=recordStep)
                        res = {
                            'lr': lr,
                            'hid1': h1,
                            'hid2': h2,
                            'actType': actType,
                            'regBeta': reg,
                            'valAcc': trainer.bestValAcc,
                            'trainLoss': trainer.lossTrainList,
                            'valLoss': trainer.lossValList,
                            'trainAcc': trainer.accTrainList,
                            'valAccList': trainer.accValList
                        }
                        self.resultList.append(res)
                        if trainer.bestValAcc > bestValAcc:
                            bestValAcc = trainer.bestValAcc
                            bestParams = {
                                'lr': lr,
                                'hid1': h1,
                                'hid2': h2,
                                'actType': actType,
                                'regBeta': reg,
                                'valAcc': trainer.bestValAcc
                            }
        tTotal = time.time() - tStart
        if verbose:
            print(f"\nGrid search finished! Total time: {tTotal:.2f}s")
            print(f"Best params: {bestParams}")
            print(f"Best val ACC: {bestValAcc:.4f}")
        self.saveRes()
        return bestParams 