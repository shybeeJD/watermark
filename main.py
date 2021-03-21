# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
# Press the green button in the gutter to run the script.


class PicDataset(Dataset):
    def __init__(self,img_path):
        self.path=img_path
        self.pic_list=[]
        for filename in os.listdir(self.path):
            if filename.endswith(".png"):
                self.pic_list.append(os.path.join(self.path,filename))


    def __getitem__(self, item):
        img=cv2.imread(self.pic_list[item],cv2.IMREAD_GRAYSCALE)
        data=np.resize(img,(1,28,28))
        data=data/255

        label=int(self.pic_list[item].split("_")[0].split("/")[-1])
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

        self.fc1 = nn.Sequential(nn.Linear(16 * 5 * 5, 120),
                                 nn.BatchNorm1d(120), nn.ReLU())

        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),
            nn.ReLU(),
            nn.Linear(84, 10))



    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LR = 0.001
torch.set_default_tensor_type(torch.DoubleTensor)
net = LeNet().to(device)
net=net.double()
# 损失函数使用交叉熵
criterion = nn.CrossEntropyLoss()
# 优化函数使用 Adam 自适应优化算法
optimizer = optim.Adam(
    net.parameters(),
    lr=LR,
)

epoch = 1
if __name__ == '__main__':
    train_dataset = PicDataset("./pic/train")
    # 下载测试集
    test_dataset = datasets.MNIST(root='./num/',
                                  train=False,
                                  transform=transforms.ToTensor(),
                                  download=True)
    batch_size = 64
    # 建立一个数据迭代器
    # 装载训练集
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    # 装载测试集
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=True)

    for epoch in range(epoch):
        sum_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            print(inputs.shape)

            optimizer.zero_grad()  # 将梯度归零
            outputs = net(inputs)  # 将数据传入网络进行前向运算

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
        output_test = net(images)
        _, predicted = torch.max(output_test, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print("correct1: ", correct)
    print("Test acc: {0}".format(correct.item() /
                                 len(test_dataset)))


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
