import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader


def get_data(path):
    data_file = pd.read_excel(path)
    data = data_file.iloc[:, :2]
    label = data_file.iloc[:, 2:3]
    # print(data, label)
    return data, label


def normalization(x, y):
    mm_x = MinMaxScaler()
    mm_y = MinMaxScaler()
    data = mm_x.fit_transform(x)
    label = mm_y.fit_transform(y)
    # print(data)
    # print(mm_y)
    return data, label, mm_y


def split_windows(data, seq_len):
    x = []
    y = []
    for i in range(len(data) - seq_len - 1):
        _x = data[i:(i+seq_len), :]
        _y = label[i + seq_len, -1]
        x.append(_x)
        y.append(_y)
    x, y = np.array(x), np.array(y)
    return x, y


def split_data(data, label, split_ratio):
    train_size = int(len(data) * split_ratio)

    data = torch.Tensor(data)
    label = torch.Tensor(label)

    train_data = data[:train_size]
    test_data = data[train_size:]

    train_label = label[:train_size]
    test_label = label[train_size:]
    # print(train_data.shape, test_data.shape)
    # print(train_label.shape, test_label.shape)
    return data, label, train_data, test_data, train_label, test_label


def data_loader(train_data, test_data, train_label, test_label, n_iters, batch_size):
    num_epochs = n_iters / (len(train_data) / batch_size)
    num_epochs = int(num_epochs)

    x_train = TensorDataset(train_data, train_label)
    x_test = TensorDataset(test_data, test_label)

    train_loader = DataLoader(x_train, batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(x_test, batch_size, shuffle=False, drop_last=True)
    # print(num_epochs)
    return train_loader, test_loader, num_epochs


class Net(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size, num_layers, output_size, batch_size, seq_len):
        super(Net, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_directions = 1

        self.lstm = nn.LSTM(out_channels, hidden_size, num_layers, batch_first=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(self.in_channels, self.out_channels, kernel_size=2)
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        self.drop_out = nn.Dropout(0.2)

    def forward(self, x):

        x = self.conv1(x)           # x : [batch_size, output_size, ]
        x = self.relu(x)
        x = x.permute(0, 2, 1)

        batch_size, seq_len = x.size()[0], x.size()[1]
        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size)
        c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size)

        output, _ = self.lstm(x, (h_0, c_0))
        output = self.drop_out(output)
        pred = self.fc(output)
        pred = pred[:, -1, :]
        return pred


n_iters = 5000
batch_size = 32
input_size = 3
out_channels = 9
hidden_size = 50
num_layers = 6
output_size = 1
seq_len = 3
split_ratio = 0.9
data_path = './data.xls'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(input_size, out_channels, hidden_size, num_layers, output_size, batch_size, seq_len)
loss_fn = torch.nn.MSELoss()
optimizer = Adam(model.parameters(), lr=0.001)

data, label = get_data(data_path)
data, label, mm_y = normalization(data, label)
data, label = split_windows(data, seq_len)
data, label, train_data, test_data, train_label, test_label = split_data(data, label, split_ratio)
train_loader, test_loader, num_epochs = data_loader(train_data, test_data, train_label, test_label, n_iters, batch_size)

# print(num_epochs)
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
    y_data_plot = np.reshape(y_data_plot, (-1, 1))
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