# PyTorch｜建立二分类任务的多层感知机模型
from numpy import vstack
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import Tensor
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD
from torch.nn import BCELoss
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_


# 数据集定义
class CSVDataset(Dataset):
    # 导入数据
    def __init__(self, path):
        # 导入传入路径的数据集为 Pandas DataFrame 格式
        df = read_csv(path, header=None)
        # 设置神经网络的输入与输出
        self.X = df.values[:, :-1]
        self.y = df.values[:, -1]
        # 确保输入数据是浮点数
        self.X = self.X.astype('float32')
        # 使用浮点型标签编码原输出
        self.y = LabelEncoder().fit_transform(self.y)
        self.y = self.y.astype('float32')
        self.y = self.y.reshape((len(self.y), 1))

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
    # 定义模型输入
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
        self.hidden3 = Linear(8, 1)
        xavier_uniform_(self.hidden3.weight)
        self.act3 = Sigmoid()

    # 前向传播
    def forward(self, X):
        # 输入到隐层 1
        X = self.hidden1(X)
        X = self.act1(X)
        # 隐层 2
        X = self.hidden2(X)
        X = self.act2(X)
        # 隐层 3 和输出
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
    criterion = BCELoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    # 枚举 epochs
    for epoch in range(100):
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
        actual = actual.reshape((len(actual), 1))
        # 转化为类值
        yhat = yhat.round()
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
path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/ionosphere.csv'
train_dl, test_dl = prepare_data(path)
print(len(train_dl.dataset), len(test_dl.dataset))
# 定义网络
model = MLP(34)
# 训练模型
train_model(train_dl, model)
# 评估模型
acc = evaluate_model(test_dl, model)
print('Accuracy: %.3f' % acc)
# 进行单个预测（预期类=1）
row = [1, 0, 0.99539, -0.05889, 0.85243, 0.02306, 0.83398, -0.37708, 1, 0.03760, 0.85243, -0.17755, 0.59755, -0.44945,
       0.60536, -0.38223, 0.84356, -0.38542, 0.58212, -0.32192, 0.56971, -0.29674, 0.36946, -0.47357, 0.56811, -0.51171,
       0.41078, -0.46168, 0.21266, -0.34090, 0.42267, -0.54487, 0.18641, -0.45300]
yhat = predict(row, model)
print('Predicted: %.3f (class=%d)' % (yhat, yhat.round()))