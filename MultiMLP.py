# PyTorch｜建立多分类任务的多层感知机模型
from numpy import vstack
from numpy import argmax
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Softmax
from torch.nn import Module
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_


# 数据集定义
class CSVDataset(Dataset):
    # 导入数据集
    def __init__(self, path):
        # 导入传入路径的数据集为 Pandas DataFrame 格式
        df = read_csv(path, header=None)
        # 设置神经网络的输入与输出
        self.X = df.values[:, :-1]
        self.y = df.values[:, -1]
        # 确保输入数据是浮点型
        self.X = self.X.astype('float32')
        # 使用浮点型标签编码原输出
        self.y = LabelEncoder().fit_transform(self.y)

    # 定义获得数据集长度的方法
    def __len__(self):
        return len(self.X)

    # 定义获得某一行数据的方法
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    # 在类内部定义划分训练集和测试集的方法，在本例中，训练集比例为 0.67，测试集比例为 0.33
    def get_splits(self, n_test=0.33):
        # 确定训练集和测试集的尺寸
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        # 根据尺寸划分训练集和测试集并返回
        return random_split(self, [train_size, test_size])


# 模型定义
class MLP(Module):
    # 定义模型属性
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        # 输入到隐层 1
        self.hidden1 = Linear(n_inputs, 10)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        # 隐层 2
        self.hidden2 = Linear(10, 8)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        # 隐层 3 和输出
        self.hidden3 = Linear(8, 3)
        xavier_uniform_(self.hidden3.weight)
        self.act3 = Softmax(dim=1)

    # 前向传播
    def forward(self, X):
        # 输入到隐层 1
        X = self.hidden1(X)
        X = self.act1(X)
        # 隐层 2
        X = self.hidden2(X)
        X = self.act2(X)
        # 输出层
        X = self.hidden3(X)
        X = self.act3(X)
        return X


# 准备数据集
def prepare_data(path):
    # 导入数据集
    dataset = CSVDataset(path)
    # 划分训练集和测试集并返回
    train, test = dataset.get_splits()
    # 为训练集和测试集创建 DataLoader
    train_dl = DataLoader(train, batch_size=32, shuffle=True)
    test_dl = DataLoader(test, batch_size=1024, shuffle=False)
    return train_dl, test_dl


# 训练模型
def train_model(train_dl, model):
    # 定义优化器
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    # 枚举 epochs
    for epoch in range(500):
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
        # 转换为类标签
        yhat = argmax(yhat, axis=1)
        # 为 stacking reshape 矩阵
        actual = actual.reshape((len(actual), 1))
        yhat = yhat.reshape((len(yhat), 1))
        # 保存
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # 计算准确度
    acc = accuracy_score(actuals, predictions)
    return acc


# 对一行数据进行类预测
def predict(row, model):
    # 转换源数据
    row = Tensor([row])
    # 做出预测
    yhat = model(row)
    # 转化为 numpy 数据类型
    yhat = yhat.detach().numpy()
    return yhat


# 准备数据
path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv'
train_dl, test_dl = prepare_data(path)
print(len(train_dl.dataset), len(test_dl.dataset))
# 定义网络
model = MLP(4)
# 训练模型
train_model(train_dl, model)
# 评估模型
acc = evaluate_model(test_dl, model)
print('Accuracy: %.3f' % acc)
# 进行单个预测
row = [5.1, 3.5, 1.4, 0.2]
yhat = predict(row, model)
print('Predicted: %s (class=%d)' % (yhat, argmax(yhat)))