import numpy as np
import pickle

class ActFuncBase:
    """激活函数抽象基类"""
    def forward(self, inputArr):
        raise NotImplementedError
    def backward(self, inputArr, gradOut):
        raise NotImplementedError

class SigmoidFunc(ActFuncBase):
    """Sigmoid激活"""
    def forward(self, inputArr):
        return 1.0 / (1.0 + np.exp(-inputArr))
    def backward(self, inputArr, gradOut):
        sig = self.forward(inputArr)
        return gradOut * sig * (1 - sig)

class ReluFunc(ActFuncBase):
    """ReLU激活"""
    def forward(self, inputArr):
        return np.maximum(0, inputArr)
    def backward(self, inputArr, gradOut):
        grad = gradOut.copy()
        grad[inputArr <= 0] = 0
        return grad

class TanhFunc(ActFuncBase):
    """Tanh激活"""
    def forward(self, inputArr):
        return np.tanh(inputArr)
    def backward(self, inputArr, gradOut):
        tanhVal = self.forward(inputArr)
        return gradOut * (1 - tanhVal * tanhVal)

class SoftmaxCrossEntropy:
    """Softmax+交叉熵损失"""
    def forward(self, logits, labels):
        self.labels = labels
        self.shape = logits.shape
        logitsAdj = logits - np.max(logits, axis=1, keepdims=True)
        expVals = np.exp(logitsAdj)
        self.softmax = expVals / np.sum(expVals, axis=1, keepdims=True)
        if self.labels.ndim == 1:
            self.labelsOneHot = np.zeros_like(self.softmax)
            self.labelsOneHot[np.arange(self.labels.shape[0]), self.labels] = 1
        else:
            self.labelsOneHot = self.labels
        lossVal = -np.sum(self.labelsOneHot * np.log(self.softmax + 1e-10)) / self.shape[0]
        return lossVal
    def backward(self):
        grad = (self.softmax - self.labelsOneHot) / self.shape[0]
        return grad

class ThreeLayerClassifier:
    def __init__(self, inDim, hidDim1, hidDim2, outDim, actType='relu', wInitStd=0.01):
        self.paramDict = {}
        self.paramDict['W1'] = wInitStd * np.random.randn(inDim, hidDim1)
        self.paramDict['b1'] = np.zeros(hidDim1)
        self.paramDict['W2'] = wInitStd * np.random.randn(hidDim1, hidDim2)
        self.paramDict['b2'] = np.zeros(hidDim2)
        self.paramDict['W3'] = wInitStd * np.random.randn(hidDim2, outDim)
        self.paramDict['b3'] = np.zeros(outDim)
        if actType.lower() == 'relu':
            self.actFunc = ReluFunc()
        elif actType.lower() == 'sigmoid':
            self.actFunc = SigmoidFunc()
        elif actType.lower() == 'tanh':
            self.actFunc = TanhFunc()
        else:
            raise ValueError(f"不支持的激活类型: {actType}")
        self.lossObj = SoftmaxCrossEntropy()
        self.inputArr = None
        self.layer1Z = None
        self.layer1A = None
        self.layer2Z = None
        self.layer2A = None
        self.layer3Z = None
    def predict(self, inputArr):
        W1, W2, W3 = self.paramDict['W1'], self.paramDict['W2'], self.paramDict['W3']
        b1, b2, b3 = self.paramDict['b1'], self.paramDict['b2'], self.paramDict['b3']
        z1 = np.dot(inputArr, W1) + b1
        a1 = self.actFunc.forward(z1)
        z2 = np.dot(a1, W2) + b2
        a2 = self.actFunc.forward(z2)
        z3 = np.dot(a2, W3) + b3
        return z3
    def loss(self, inputArr, labels, regBeta=0.0):
        logits = self.predict(inputArr)
        lossVal = self.lossObj.forward(logits, labels)
        if regBeta > 0:
            W1, W2, W3 = self.paramDict['W1'], self.paramDict['W2'], self.paramDict['W3']
            regLoss = 0.5 * regBeta * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2))
            lossVal += regLoss
        return lossVal
    def accuracy(self, inputArr, labels):
        logits = self.predict(inputArr)
        predIdx = np.argmax(logits, axis=1)
        if labels.ndim != 1:
            labels = np.argmax(labels, axis=1)
        accVal = np.sum(predIdx == labels) / float(inputArr.shape[0])
        return accVal
    def forward(self, inputArr):
        self.inputArr = inputArr
        W1, W2, W3 = self.paramDict['W1'], self.paramDict['W2'], self.paramDict['W3']
        b1, b2, b3 = self.paramDict['b1'], self.paramDict['b2'], self.paramDict['b3']
        self.layer1Z = np.dot(inputArr, W1) + b1
        self.layer1A = self.actFunc.forward(self.layer1Z)
        self.layer2Z = np.dot(self.layer1A, W2) + b2
        self.layer2A = self.actFunc.forward(self.layer2Z)
        self.layer3Z = np.dot(self.layer2A, W3) + b3
        return self.layer3Z
    def backward(self, labels, regBeta=0.0):
        batchSz = self.inputArr.shape[0]
        gradOut = self.lossObj.backward()
        gradW3 = np.dot(self.layer2A.T, gradOut)
        gradb3 = np.sum(gradOut, axis=0)
        gradA2 = np.dot(gradOut, self.paramDict['W3'].T)
        gradZ2 = self.actFunc.backward(self.layer2Z, gradA2)
        gradW2 = np.dot(self.layer1A.T, gradZ2)
        gradb2 = np.sum(gradZ2, axis=0)
        gradA1 = np.dot(gradZ2, self.paramDict['W2'].T)
        gradZ1 = self.actFunc.backward(self.layer1Z, gradA1)
        gradW1 = np.dot(self.inputArr.T, gradZ1)
        gradb1 = np.sum(gradZ1, axis=0)
        if regBeta > 0:
            gradW1 += regBeta * self.paramDict['W1']
            gradW2 += regBeta * self.paramDict['W2']
            gradW3 += regBeta * self.paramDict['W3']
        gradDict = {}
        gradDict['W1'] = gradW1
        gradDict['b1'] = gradb1
        gradDict['W2'] = gradW2
        gradDict['b2'] = gradb2
        gradDict['W3'] = gradW3
        gradDict['b3'] = gradb3
        return gradDict
    def saveParams(self, filePath):
        with open(filePath, 'wb') as f:
            pickle.dump(self.paramDict, f)
    def loadParams(self, filePath):
        with open(filePath, 'rb') as f:
            self.paramDict = pickle.load(f) 