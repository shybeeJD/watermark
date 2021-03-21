import torch
import torch.nn as nn
import os
import cv2
from lstm import GRUNet
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch.optim as optim

class PicDataset(Dataset):
    def __init__(self,img_path):
        self.path=img_path
        self.pic_list=[]
        for filename in os.listdir(self.path):
            if filename.endswith(".png"):
                self.pic_list.append(os.path.join(self.path,filename))


    def __getitem__(self, item):
        img=cv2.imread(self.pic_list[item],cv2.IMREAD_GRAYSCALE)
        data=np.resize(img,(1,48,28))
        data=data/255

        label=int(self.pic_list[item].split("_")[1])
        return data,label
    def __len__(self):
        return len(self.pic_list)

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 6, 3, 1, 2), nn.ReLU(),
                                   nn.MaxPool2d(2, 2))

        self.conv2 = nn.Sequential(nn.Conv2d(6, 16, 5), nn.ReLU(),
                                   nn.MaxPool2d(2, 2))

        self.fc1 = nn.Sequential(nn.Linear(16 * 5 * 5, 128),
                                 nn.BatchNorm1d(128), nn.ReLU())

        self.fc2 = nn.Sequential(
            nn.Linear(128, 84),
            nn.BatchNorm1d(84),
            nn.ReLU(),
            nn.Linear(84, 10))



    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        self.featuremap = x.detach()
        x = self.fc2(x)
        return x,self.featuremap

class WaterMarkedNet(nn.Module):
    def __init__(self):
        super(WaterMarkedNet, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lenet=LeNet().to(device)

        self.gruNet=GRUNet(1).to(device)
        self.gruNet.load_state_dict(torch.load("models/lstmnet_gru_4000.pth"))
        self.gruNet.requires_grad_(False)
        self.a=nn.Linear(128,256)
        self.b=nn.Linear(128,256)
        self.c=nn.Linear(64,10)
        self.fc2 = nn.Sequential(
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 84),
            nn.BatchNorm1d(84),
            nn.ReLU(),
            nn.Linear(84, 10))


    def forward(self, x):
        t,x1=self.lenet(x[:,:,0:28,:])
        tmp=x[:,:,28:48,0]
        tmp=tmp.view(-1,20,1)
        y2,x2=self.gruNet(tmp)
        y1=self.a(x1)+self.b(x2)
        y1=self.fc2(y1)

        return y1,y2

def train():
    batch_size=8
    epoch=15
    dataset = PicDataset("./pic/trigger")
    train_size = int(len(dataset) * 0.7)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    LR = 0.001
    torch.set_default_tensor_type(torch.DoubleTensor)
    net = WaterMarkedNet().to(device)
    net = net.double()
    # 损失函数使用交叉熵
    criterion = nn.CrossEntropyLoss()
    # 优化函数使用 Adam 自适应优化算法
    optimizer = optim.Adam(
        net.parameters(),
        lr=LR,
    )
    for epoch in range(epoch):
        sum_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            if inputs.shape[0]!=batch_size:
                continue

            optimizer.zero_grad()  # 将梯度归零
            outputs,mark = net(inputs)  # 将数据传入网络进行前向运算

            loss = criterion(outputs, labels)  # 得到损失函数
            loss.backward()  # 反向传播
            optimizer.step()  # 通过梯度做一步参数更新

            # print(loss)
            sum_loss += loss.item()
            if i % 100 == 99:
                print('[%d,%d] loss:%.03f' %
                      (epoch + 1, i + 1, sum_loss / 100))
                sum_loss = 0.0
        net.eval()  # 将模型变换为测试模式
        correct = 0
        total = 0
        for data_test in test_loader:
            images, labels = data_test
            images=images.double()
            output_test,mark = net(images)
            _, predicted = torch.max(output_test, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        print("correct1: ", correct)
        print("Test acc: {0}".format(correct.item() /
                                 len(test_dataset)))
    torch.save(obj=net.state_dict(), f="models/MNIST_1000.pth")
if __name__ == '__main__':
    test_dataset = PicDataset("./pic/triggertest")
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=100,
                                               shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    LR = 0.001
    torch.set_default_tensor_type(torch.DoubleTensor)
    net = WaterMarkedNet().to(device)
    net = net.double()
    net.load_state_dict(torch.load("models/MNIST_1000.pth"))
    net.eval()  # 将模型变换为测试模式
    correct = 0
    total = 0
    for data_test in test_loader:
        images, labels = data_test
        images = images.double()
        output_test, mark = net(images)
        _, predicted = torch.max(output_test, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print("correct1: ", correct)
    print("Test acc: {0}".format(correct.item() /
                                 len(test_dataset)))