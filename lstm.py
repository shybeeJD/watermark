import os

import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
import csv
import torch.nn.functional as F

class Lstm(nn.Module):
    def __init__(self,input_size=1):
        super(Lstm, self).__init__()
        self.lstm=nn.LSTM(input_size=input_size, hidden_size=128, num_layers=2)

        self.fc1 = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 8))

    def forward(self, x):
        x, (hn, cn) = self.lstm(x)
        #x = x.view(x.size()[0]*x.size()[1],-1)
        x=x[:,-1,:]
        x=self.fc1(x)
        return x


class GRUNet(nn.Module):

    def __init__(self, input_size):
        super(GRUNet, self).__init__()
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.layer1 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            #nn.Linear(128, 8)
        )
        self.layer2=nn.Linear(128, 9)

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)  # None 表示 hidden state 会用全0的 state
        out = self.layer1(r_out[:, -1])
        self.featuremap = out.detach()
        out=self.layer2(out)
        return out,self.featuremap


class SqueDataset(Dataset):
    def __init__(self,Sque_path):
        self.path=Sque_path
        self.Sque_list=[]
        with open(self.path)as f:
            f_csv = csv.reader(f)
            for row in f_csv:
                row=[int(i) for i in row]
                self.Sque_list.append(row)

    def __getitem__(self, item):

        data=self.Sque_list[item][0:len(self.Sque_list[item])-1]
        data=np.array(data)
        data=np.resize(data,(data.shape[0],1))
        label=self.Sque_list[item][-1]
        return data/8,np.array(label)
    def __len__(self):
        return len(self.Sque_list)



'''
data=np.array(data)
print(data.shape)
print(data[0,:])
data=np.swapaxes(data,0,1)
data=np.reshape(data,(data.shape[0],data.shape[1],1))
print(data.shape)
test=data[:,0:8,:]
'''



def train():
    epoch=120
    batch_size = 4
    loss_F = torch.nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = GRUNet(1).to(device)
    net.double()
    LR = 0.0001
    criterion = nn.CrossEntropyLoss()
    # 优化函数使用 Adam 自适应优化算法
    optimizer = optim.Adam(
        net.parameters(),
        lr=LR,
    )
    trainData = SqueDataset("data.csv")
    train_size = int(len(trainData) * 0.7)
    test_size = len(trainData) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(trainData, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(dataset=trainData,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_loader=torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=batch_size*50,
                                               shuffle=True)

    for ep in range(epoch):
        for i, data in enumerate(train_loader):
            x, y = data
            pred,mark = net(x)
            loss = loss_F(pred, y)  # 计算loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 50 == 49:  # 每50步，计算精度
                print('Train Epoch: {}\t Loss: {:.6f}'.format(ep, loss.item()))
        with torch.no_grad():
            for j, test in enumerate(test_loader):
                test_x, test_y = test
                test_pred,mark = net(test_x)
                prob = torch.nn.functional.softmax(test_pred, dim=1)
                pred_cls = torch.argmax(prob, dim=1)
                print(len(pred_cls), len(test_y))
                acc = (pred_cls == test_y).sum().numpy() / pred_cls.size()[0]
                print(f"{epoch}-{i}: accuracy:{acc}")


    torch.save(obj=net.state_dict(), f="models/gru_5000_noisy.pth")



def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = GRUNet(1).to(device)
    net.double()
    net.load_state_dict(torch.load("models/lstmnet_gru_3000.pth"))
    testData = SqueDataset("test.csv")
    test_loader = torch.utils.data.DataLoader(dataset=testData,
                                               batch_size=100,
                                               shuffle=True)
    with torch.no_grad():
        for j, test in enumerate(test_loader):
            test_x, test_y = test
            test_pred = net(test_x)
            prob = torch.nn.functional.softmax(test_pred, dim=1)
            pred_cls = torch.argmax(prob, dim=1)
            print(len(pred_cls), len(test_y))
            acc = (pred_cls == test_y).sum().numpy() / pred_cls.size()[0]
            print(f"{0}-{j}: accuracy:{acc}")

if __name__ == "__main__":
    train()