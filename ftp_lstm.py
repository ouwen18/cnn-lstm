import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.nn import LSTM, Linear, Dropout
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader


# 获取数据
def get_data(path):
    data_file = pd.read_excel(path)
    data = data_file.iloc[:, :2]
    label = data_file.iloc[:, 2:3]
    # print(data)
    # print(label)
    return data, label


# 数据预处理
def normalization(data, label):
    mm_x = MinMaxScaler()           # 正则化
    mm_y = MinMaxScaler()
    data = mm_x.fit_transform(data)
    label = mm_y.fit_transform(label)
    # print(data)
    return data, label, mm_y


# 时间向量转换
def split_windows(data, label, seq_len):
    x = []
    y = []
    for i in range(len(data) - seq_len - 1):
        _x = data[i:(i+seq_len), :]
        _y = label[i+seq_len, -1]
        x.append(_x)
        y.append(_y)
    x, y = np.array(x), np.array(y)
    # print(x.shape, y.shape)
    return x, y


# 划分数据集
def split_data(data, label, split_ratio):
    train_size = int(len(data) * split_ratio)
    test_size = int(len(data) - train_size)

    data = torch.Tensor(data)
    label = torch.Tensor(label)

    train_data = torch.Tensor(data[:train_size])
    test_data = torch.Tensor(data[train_size:len(data)])

    train_label = torch.Tensor(label[:train_size])
    test_label = torch.Tensor(label[train_size:len(label)])

    return data, label, train_data, train_label, test_data, test_label


# 数据加载
def data_loader(x_train, y_train, x_test, y_test, n_iters, batch_size):
    num_epochs = n_iters / (len(x_train) / batch_size)
    num_epochs = int(num_epochs)

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)

    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=False, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False, drop_last=True)

    return train_dataloader, test_dataloader, num_epochs


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size, seq_len):
        super(Net, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_directions = 1

        self.lstm = LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.drop_out = Dropout(0.2)
        self.fc = Linear(hidden_size, output_size)

    def forward(self, input):
        batch_size, seq_len = input.size()[0], input.size()[1]

        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size)
        c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size)

        input = input.permute(0, 2, 1)
        input = torch.Tensor(input)
        output, _ = self.lstm(input, (h_0, c_0))
        output = self.drop_out(output)
        pred = self.fc(output)
        pred = pred[:, -1, :]
        return pred


data_path = './data.xls'
input_size = 3
seq_len = 3
num_layers = 6
hidden_size = 50
batch_size = 64
n_iters = 5000
output_size = 1
split_radio = 0.9
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(input_size, hidden_size, num_layers, output_size, batch_size, seq_len)
loss_fn = torch.nn.MSELoss()
optimizer = Adam(model.parameters(), lr=0.001)


data, label = get_data(data_path)
data, label, mm_y = normalization(data, label)
x, y = split_windows(data, label, seq_len)
data, label, train_data, train_label, test_data, test_label = split_data(x, y, split_radio)
train_loader, test_loader, num_epochs = data_loader(train_data, train_label, test_data, test_label, n_iters, batch_size)


iter = 0
for epoch in range(num_epochs):
    for i, (data, label) in enumerate(train_loader):
        # print(data.shape)
        pred = model(data)

        loss = loss_fn(pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        iter += 1
        if iter % 100 == 0:
            print("iter：{}，loss：{}".format(iter, loss.item()))


def result(x_data, y_data):
    model.eval()

    x_data = x_data
    y_data = y_data
    train_predict = model(x_data)

    data_predict = train_predict.data.numpy()
    y_data_plot = y_data.data.numpy()
    y_data_plot = np.reshape(y_data_plot, (-1,1))
    data_predict = mm_y.inverse_transform(data_predict)
    y_data_plot = mm_y.inverse_transform(y_data_plot)

    plt.plot(y_data_plot)
    plt.plot(data_predict)
    plt.legend(('real', 'predict'), fontsize='15')
    plt.show()

    print('MAE/RMSE')
    print(mean_absolute_error(y_data_plot, data_predict))
    print(np.sqrt(mean_squared_error(y_data_plot, data_predict)))


result(data, label)
result(test_data, test_label)
