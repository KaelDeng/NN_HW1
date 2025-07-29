import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from model import ThreeLayerClassifier
from data import load_cifar10

def runTest(modelFile, dataDir='./cifar10/dataset', inDim=3072, hid1=100, hid2=100, outDim=10, actType='relu'):
    """
    测试模型在测试集上的表现
    """
    print("Loading CIFAR-10 dataset...")
    _, (Xtest, Ytest) = load_cifar10(dataDir, normalize=True, flatten=True, one_hot=False)
    print(f"Building model (hid1={hid1}, hid2={hid2}, act={actType})...")
    net = ThreeLayerClassifier(
        inDim=inDim,
        hidDim1=hid1,
        hidDim2=hid2,
        outDim=outDim,
        actType=actType
    )
    print(f"Loading parameters from {modelFile} ...")
    net.loadParams(modelFile)
    print("Evaluating model...")
    acc = net.accuracy(Xtest, Ytest)
    print(f"Test accuracy: {acc:.4f}")
    showPred(net, Xtest, Ytest)
    return acc

def showPred(net, X, y, nShow=10, savePath='./cifar10/results/test_pred.png'):
    """
    可视化模型预测
    """
    classNames = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    idx = np.random.choice(X.shape[0], nShow, replace=False)
    Xs = X[idx]
    ys = y[idx]
    pred = net.predict(Xs)
    predCls = np.argmax(pred, axis=1)
    plt.figure(figsize=(12, 4))
    for i in range(nShow):
        plt.subplot(2, 5, i+1)
        img = Xs[i].reshape(32, 32, 3)
        plt.imshow(img)
        plt.title(f"True: {classNames[ys[i]]}\nPred: {classNames[predCls[i]]}")
        plt.axis('off')
    plt.tight_layout()
    os.makedirs(os.path.dirname(savePath), exist_ok=True)
    plt.savefig(savePath)
    plt.close()
    print(f"Prediction visualization saved to {savePath}")

def showConfMat(net, X, y, savePath='./cifar10/results/confmat.png'):
    """
    计算并可视化混淆矩阵
    """
    classNames = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    pred = net.predict(X)
    predCls = np.argmax(pred, axis=1)
    confMat = np.zeros((10, 10), dtype=int)
    for t, p in zip(y, predCls):
        confMat[t, p] += 1
    plt.figure(figsize=(10, 8))
    plt.imshow(confMat, interpolation='nearest', cmap=plt.cm.RdBu)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, classNames, rotation=45)
    plt.yticks(tick_marks, classNames)
    thresh = confMat.max() / 2.
    for i in range(confMat.shape[0]):
        for j in range(confMat.shape[1]):
            plt.text(j, i, confMat[i, j],
                    ha="center", va="center",
                    color="white" if confMat[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    os.makedirs(os.path.dirname(savePath), exist_ok=True)
    plt.savefig(savePath)
    plt.close()
    print(f"Confusion matrix saved to {savePath}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='测试神经网络在CIFAR-10数据集上的性能')
    parser.add_argument('--model_path', type=str, default='./cifar10/results/best_model.pkl', help='模型参数文件路径')
    parser.add_argument('--dataset_dir', type=str, default='./cifar10/dataset', help='数据集目录')
    parser.add_argument('--hidden_size1', type=int, default=100, help='第一隐藏层大小')
    parser.add_argument('--hidden_size2', type=int, default=100, help='第二隐藏层大小')
    parser.add_argument('--activation', type=str, default='relu', help='激活函数类型 (relu, sigmoid, tanh)')
    args = parser.parse_args()
    _, (Xtest, Ytest) = load_cifar10(args.dataset_dir, normalize=True, flatten=True, one_hot=False)
    net = ThreeLayerClassifier(
        inDim=Xtest.shape[1],
        hidDim1=args.hidden_size1,
        hidDim2=args.hidden_size2,
        outDim=10,
        actType=args.activation
    )
    net.loadParams(args.model_path)
    acc = net.accuracy(Xtest, Ytest)
    print(f"Test accuracy: {acc:.4f}")
    showPred(net, Xtest, Ytest)
    showConfMat(net, Xtest, Ytest) 