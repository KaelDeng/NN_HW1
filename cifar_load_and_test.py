import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from model import ThreeLayerClassifier
from data import load_cifar10, DataSplitter
from trainer import SgdOpt, LrScheduler, NetTrainer
from hyperparameter_search import ParamSearcher
from test import runTest

def main():
    parser = argparse.ArgumentParser(description='CIFAR-10三层神经网络分类器')
    # 数据集参数
    parser.add_argument('--dataset_dir', type=str, default='./cifar10', help='数据集保存目录')
    parser.add_argument('--val_size', type=float, default=0.1, help='验证集比例')
    # 模型参数
    parser.add_argument('--hidden_size1', type=int, default=100, help='第一隐藏层大小')
    parser.add_argument('--hidden_size2', type=int, default=100, help='第二隐藏层大小')
    parser.add_argument('--activation', type=str, default='relu', help='激活函数类型 (relu, sigmoid, tanh)')
    # 训练参数
    parser.add_argument('--lr', type=float, default=0.01, help='学习率')
    parser.add_argument('--batch_size', type=int, default=100, help='批大小')
    parser.add_argument('--epochs', type=int, default=300, help='训练轮数')
    parser.add_argument('--reg_lambda', type=float, default=0.01, help='L2正则化系数')
    parser.add_argument('--lr_decay', action='store_true', help='是否使用学习率衰减')
    parser.add_argument('--decay_rate', type=float, default=0.95, help='学习率衰减率')
    # 模式选择
    parser.add_argument('--mode', type=str, default='train', help='运行模式 (train, search, test)')
    parser.add_argument('--search_type', type=str, default='all', help='搜索类型 (lr, hidden, reg, activation, all)')
    parser.add_argument('--model_path', type=str, default='./cifar10/results/best_model.pkl', help='测试模式下的模型路径')
    # 其他参数
    parser.add_argument('--save_dir', type=str, default='./cifar10/results', help='结果保存目录')
    parser.add_argument('--record_interval', type=int, default=1, help='记录训练过程的间隔')
    args = parser.parse_args()
    # 加载和处理数据
    print("Loading CIFAR-10 dataset...")
    (XtrFull, YtrFull), (Xtest, Ytest) = load_cifar10(dataset_dir=args.dataset_dir, normalize=True, flatten=True, one_hot=False)
    print(f"Splitting train/val set (val ratio: {args.val_size})...")
    Xtr, Xval, Ytr, Yval = DataSplitter.train_val_split(XtrFull, YtrFull, val_size=args.val_size)
    print(f"Dataset size - train: {Xtr.shape[0]}, val: {Xval.shape[0]}, test: {Xtest.shape[0]}")
    if args.mode == 'train':
        trainMain(args, Xtr, Ytr, Xval, Yval)
    elif args.mode == 'search':
        searchMain(args, Xtr, Ytr, Xval, Yval)
    elif args.mode == 'test':
        testMain(args, Xtest, Ytest)
    else:
        print(f"Unsupported mode: {args.mode}")

def trainMain(args, Xtr, Ytr, Xval, Yval):
    print("\nStart training model...")
    net = ThreeLayerClassifier(
        inDim=Xtr.shape[1],
        hidDim1=args.hidden_size1,
        hidDim2=args.hidden_size2,
        outDim=10,
        actType=args.activation
    )
    opt = SgdOpt(lr=args.lr)
    lrSched = None
    if args.lr_decay:
        lrSched = lambda initLr, ep: LrScheduler.expDecay(initLr, ep, decay=args.decay_rate)
    trainer = NetTrainer(net, opt, lrSched=lrSched, savePath=args.save_dir)
    trainer.train(Xtr, Ytr, Xval, Yval, nEpoch=args.epochs, batchSz=args.batch_size, regBeta=args.reg_lambda, recordStep=args.record_interval)
    trainer.plotCurve()
    trainer.showWeights(whichLayer=1, reshape=True)
    trainer.showWeights(whichLayer=2)
    trainer.showWeights(whichLayer=3)
    print(f"Training finished! Best val ACC: {trainer.bestValAcc:.4f}")
    print(f"Model saved to {os.path.join(args.save_dir, 'best_model.pkl')}")

def searchMain(args, Xtr, Ytr, Xval, Yval):
    print("\nStart hyperparameter search...")
    searcher = ParamSearcher(savePath=os.path.join(args.save_dir, 'hyperparam_search'))
    if args.search_type == 'lr' or args.search_type == 'all':
        lrList = [0.001, 0.01, 0.05, 0.1, 0.5]
        print(f"\nSearching learning rate: {lrList}")
        bestLr = searcher.searchLr(Xtr, Ytr, Xval, Yval, hid1=args.hidden_size1, hid2=args.hidden_size2, lrList=lrList, actType=args.activation, batchSz=args.batch_size, nEpoch=int(args.epochs/2), regBeta=args.reg_lambda, recordStep=args.record_interval)
        print(f"Best learning rate: {bestLr}")
        if args.search_type == 'all':
            args.lr = bestLr
    if args.search_type == 'hidden' or args.search_type == 'all':
        print("\nSearching hidden layer size...")
        hiddenList = [(50, 50), (100, 100), (200, 100), (300, 150), (500, 200)]
        bestHidden = searcher.searchHidden(Xtr, Ytr, Xval, Yval, hiddenList=hiddenList, lr=args.lr, actType=args.activation, batchSz=args.batch_size, nEpoch=int(args.epochs/2), regBeta=args.reg_lambda, recordStep=args.record_interval)
        print(f"Best hidden size: {bestHidden}")
        if args.search_type == 'all':
            args.hidden_size1 = bestHidden[0]
            args.hidden_size2 = bestHidden[1]
    if args.search_type == 'reg' or args.search_type == 'all':
        print("\nSearching regularization...")
        regList = [0.0, 0.0001, 0.001, 0.01, 0.1]
        bestReg = searcher.searchReg(Xtr, Ytr, Xval, Yval, hid1=args.hidden_size1, hid2=args.hidden_size2, regList=regList, lr=args.lr, actType=args.activation, batchSz=args.batch_size, nEpoch=int(args.epochs/2), recordStep=args.record_interval)
        print(f"Best regularization: {bestReg}")
        if args.search_type == 'all':
            args.reg_lambda = bestReg
    if args.search_type == 'activation' or args.search_type == 'all':
        print("\nSearching activation function...")
        actList = ['relu', 'sigmoid', 'tanh']
        bestAct = searcher.searchAct(Xtr, Ytr, Xval, Yval, hid1=args.hidden_size1, hid2=args.hidden_size2, actList=actList, lr=args.lr, batchSz=args.batch_size, nEpoch=int(args.epochs/2), regBeta=args.reg_lambda, recordStep=args.record_interval)
        print(f"Best activation function: {bestAct}")
        if args.search_type == 'all':
            args.activation = bestAct
    if args.search_type == 'all':
        print("\nTraining with best hyperparameters...")
        print(f"Best hyperparameters - lr: {args.lr}, hidden: ({args.hidden_size1}, {args.hidden_size2}), reg: {args.reg_lambda}, act: {args.activation}")
        trainMain(args, Xtr, Ytr, Xval, Yval)
    searcher.saveRes()
    print(f"Hyperparameter search results saved to {os.path.join(searcher.savePath, 'search_res.pkl')}")

def testMain(args, Xtest, Ytest):
    print("\nStart testing model...")
    acc = runTest(
        modelFile=args.model_path,
        dataDir=args.dataset_dir,
        inDim=Xtest.shape[1],
        hid1=args.hidden_size1,
        hid2=args.hidden_size2,
        actType=args.activation
    )
    print(f"Testing finished! Test accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
