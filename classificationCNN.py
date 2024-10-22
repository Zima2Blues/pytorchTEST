# PyTorch｜建立图像分类的卷积神经网络模型
from numpy import vstack
from numpy import argmax
from pandas import read_csv
from sklearn.metrics import accuracy_score
from torchvision.datasets import MNIST
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor
from torchvision.transforms import Normalize
from torch.utils.data import DataLoader
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Softmax
from torch.nn import Module
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_


# 模型定义
class CNN(Module):
    # 定义模型属性
    def __init__(self, n_channels):
        super(CNN, self).__init__()
        # 输入到隐层 1
        self.hidden1 = Conv2d(n_channels, 32, (3, 3))
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        # 池化层 1
        self.pool1 = MaxPool2d((2, 2), stride=(2, 2))
        # 隐层 2
        self.hidden2 = Conv2d(32, 32, (3, 3))
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        # 池化层 2
        self.pool2 = MaxPool2d((2, 2), stride=(2, 2))
        # 全连接层
        self.hidden3 = Linear(5 * 5 * 32, 100)
        kaiming_uniform_(self.hidden3.weight, nonlinearity='relu')
        self.act3 = ReLU()
        # 输出层
        self.hidden4 = Linear(100, 10)
        xavier_uniform_(self.hidden4.weight)
        self.act4 = Softmax(dim=1)

    # 前向传播
    def forward(self, X):
        # 输入到隐层 1
        X = self.hidden1(X)
        X = self.act1(X)
        X = self.pool1(X)
        # 隐层 2
        X = self.hidden2(X)
        X = self.act2(X)
        X = self.pool2(X)
        # 扁平化
        X = X.view(-1, 4 * 4 * 50)
        # 隐层 3
        X = self.hidden3(X)
        X = self.act3(X)
        # 输出层
        X = self.hidden4(X)
        X = self.act4(X)
        return X


# 准备数据集
def prepare_data(path):
    # 定义标准化
    trans = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    # 加载数据集
    train = MNIST(path, train=True, download=True, transform=trans)
    test = MNIST(path, train=False, download=True, transform=trans)
    # 为训练集和测试集创建 DataLoader
    train_dl = DataLoader(train, batch_size=64, shuffle=True)
    test_dl = DataLoader(test, batch_size=1024, shuffle=False)
    return train_dl, test_dl


# 训练模型
def train_model(train_dl, model):
    # 定义优化器
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    # 枚举 epochs
    for epoch in range(10):
        # 枚举 mini batches
        for i, (inputs, targets) in enumerate(train_dl):
            # 梯度清除
            optimizer.zero_grad()
            # 计算模型输出
            yhat = model(inputs)
            # 计算损失
            loss = criterion(yhat, targets)
            # 贡献度分配
            loss.backward()
            # 升级模型权重
            optimizer.step()


# 评估模型
def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        # 在测试集上评估模型
        yhat = model(inputs)
        # 转化为 numpy 数据类型
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        # 转化为类标签
        yhat = argmax(yhat, axis=1)
        # 为 stack 格式化数据集
        actual = actual.reshape((len(actual), 1))
        yhat = yhat.reshape((len(yhat), 1))
        # 保存
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # 计算准确度
    acc = accuracy_score(actuals, predictions)
    return acc


# 准备数据
path = '~/.torch/datasets/mnist'
train_dl, test_dl = prepare_data(path)
print(len(train_dl.dataset), len(test_dl.dataset))
# 定义网络
model = CNN(1)
# # 训练模型
train_model(train_dl, model)  # 该步骤运行约需 5 分钟。
# 评估模型
acc = evaluate_model(test_dl, model)
print('Accuracy: %.3f' % acc)