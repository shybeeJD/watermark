from collections import OrderedDict

import torch
import torch.nn as nn
import os
import cv2
from lstm import GRUNet
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch.optim as optim
import torch.nn.utils.prune as prune
from torchsummary import summary
import random

labels = [1, 5, 0, 0, 4, 5, 2, 3, 1, 2, 7, 7, 1, 1, 1, 3, 2, 6, 2, 3, 7, 7, 1, 7, 7, 6, 2, 7, 2, 1, 7, 4, 1, 2, 0,
          6, 2, 0, 5, 3, 7, 0, 4, 3, 2, 6, 4, 6, 5, 6, 6, 4, 1, 6, 4, 3, 6, 4, 4, 1, 3, 7, 3, 7, 0, 5, 4, 4, 4, 7,
          1, 4, 6, 1, 4, 3, 5, 6, 1, 5, 1, 0, 5, 1, 3, 3, 0, 4, 2, 6, 1, 0, 6, 5, 6, 3, 4, 4, 2, 3, 7, 2, 0, 7, 4,
          5, 1, 7, 7, 1, 5, 3, 1, 4, 7, 7, 1, 6, 7, 6, 0, 6, 0, 2, 5, 6, 5, 4, 3, 5, 2, 1, 0, 4, 3, 5, 2, 3, 3, 3,
          4, 5, 0, 1, 6, 5, 5, 7, 2, 1, 1, 3, 0, 3, 5, 6, 4, 1, 3, 7, 1, 6, 4, 0, 2, 4, 5, 2, 6, 6, 5, 6, 5, 0, 7,
          0, 5, 7, 6, 6, 2, 7, 3, 5, 0, 2, 7, 4, 1, 1, 2, 0, 6, 3, 1, 3, 3, 5, 4, 4, 3, 6, 5, 2, 0, 0, 4, 1, 4, 0,
          3, 0, 2, 3, 0, 3, 2, 2, 3, 2, 5, 4, 0, 2, 5, 0, 2, 1, 6, 4, 0, 6, 3, 0, 5, 0, 3, 3, 0, 2, 4, 3, 0, 6, 3,
          6, 3, 0, 0, 0, 0, 5, 1, 2, 7, 6, 0, 4, 1, 5, 6, 1, 2, 5, 5, 0, 6, 2, 3, 6, 3, 1, 6, 0, 1, 6, 1, 4, 6, 0,
          0, 7, 2, 5, 0, 1, 4, 5, 5, 7, 5, 2, 4, 0, 1, 4, 5, 2, 6, 3, 0, 4, 6, 1, 2, 3, 7, 2, 1, 5, 4, 7, 3, 3, 5,
          2, 3, 0, 6, 3, 3, 2, 5, 6, 4, 1, 5, 6, 3, 7, 4, 5, 3, 7, 4, 5, 2, 5, 6, 6, 3, 4]
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
        if self.pic_list[item].split("_")[-1]!="noisy.png":
            label = int(self.pic_list[item].split("_")[0].split("/")[-1])
        else:
            label=int(self.pic_list[item].split("_")[1])
        return data,label
    def __len__(self):
        return len(self.pic_list)

class NormalDataset(Dataset):
    def __init__(self,img_path):
        self.path=img_path
        self.pic_list=[]
        for filename in os.listdir(self.path):
            if filename.endswith(".png") and filename.split("_")[-1]=="noisy.png":
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
        self.conv1 = nn.Sequential((nn.Conv2d(1, 6, 3, 1, 2)), nn.ReLU(),
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
        self.gruNet.load_state_dict(torch.load("models/gru_5000_noisy.pth"))
        self.gruNet.requires_grad_(False)
        self.a=nn.Linear(128,256)
        self.b=nn.Linear(128,256)

        self.conv1 = nn.Sequential(nn.Conv2d(1, 6, 3, 1, 2), nn.ReLU(),
                                   nn.MaxPool2d(2, 2))

        self.conv2 = nn.Sequential(nn.Conv2d(6, 16, 5), nn.ReLU(),
                                   nn.MaxPool2d(2, 2))

        self.c = nn.Linear(16 * 5 * 5, 16* 5 * 5)
        self.d = nn.Linear(128, 16*5*5)
        self.d.requires_grad_(False)
        self.r= nn.ReLU()

        self.fc1 = nn.Sequential(nn.Linear(16 * 5 * 5, 128),
                                 nn.BatchNorm1d(128), nn.ReLU())

        self.fc2 = nn.Sequential(
            nn.Linear(128, 84),
            nn.BatchNorm1d(84),
            nn.ReLU(),
            nn.Linear(84, 10))





    def forward(self, x):
        tmp = x[:, :, 28:48, 0]
        tmp = tmp.view(-1, 20, 1)
        y2, x2 = self.gruNet(tmp)

        x1=self.conv1(x[:,:,0:28,:])
        x1=self.conv2(x1)
        x1 = x1.view(x.size()[0], -1)
        x1 = self.c(x1)+self.d(x2)

        x1=self.r(x1)
        x1=self.fc1(x1)
        y1=self.fc2(x1)

        return y1,y2

def train():
    batch_size=8
    epoch=10
    dataset = PicDataset("./pic/methods2/trigger")
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
        torch.save(obj=net.state_dict(), f="models/2MNIST_1000.pth")#准确率0.97899
    torch.save(obj=net.state_dict(), f="models/2MNIST_1000.pth")

def pune(rate):
    batch_size = 8
    epoch = 10
    dataset = PicDataset("./pic/normal")
    train_size = int(len(dataset) * 0.9)
    print(len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=True)
    marked =PicDataset("./pic/triggerAll2")

    mark_size=int(len(marked)*0.9)
    tmp_size = len(marked) - mark_size
    mark_data,tmp=torch.utils.data.random_split(marked, [mark_size, tmp_size])
    marked_loader = torch.utils.data.DataLoader(dataset=mark_data,
                                              batch_size=batch_size,
                                              shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = WaterMarkedNet().to(device)
    for name in net.lenet.conv1.state_dict():
        print(name)
    #summary(net, (1, 48, 28))
    net = net.double()
    net.load_state_dict(torch.load("models/2MNIST_2000.pth"))
    #summary(net, (1, 48, 28))
    criterion = nn.CrossEntropyLoss()
    # 优化函数使用 Adam 自适应优化算法
    optimizer = optim.Adam(
        net.parameters(),
        lr=0.001,
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
            parameters_to_prune = (

                (net.lenet.conv1[0], 'weight'),
                (net.lenet.conv2[0], 'weight'),
                (net.lenet.fc1[0], 'weight'),
                (net.lenet.fc2[0], 'weight'),
                #(net.c, 'weight'),
            )
            if epoch%100==99:
                prune.global_unstructured(
                    parameters_to_prune,
                    pruning_method=prune.L1Unstructured,
                    amount=rate,
                )


            # print(loss)
            sum_loss += loss.item()
            if i % 100 == 99:
                print('[%d,%d] loss:%.03f' %
                      (epoch + 1, i + 1, sum_loss / 100))
                sum_loss = 0.0
        net.eval()  # 将模型变换为测试模式
        correct = 0
        err = 0
        total=0
        for data_test in marked_loader:
            images, labels = data_test
            images=images.double()
            output_test,mark = net(images)
            _, predicted = torch.max(output_test, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            err=(predicted != labels).sum()
        print("Marked acc: {0}".format(correct.item() /
                                 len(mark_data)))
        correct = 0
        err=0
        total = 0
        for data_test in test_loader:
            images, labels = data_test
            images=images.double()
            output_test,mark = net(images)
            _, predicted = torch.max(output_test, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            err = (predicted != labels).sum()
        print("Test acc: {0}".format(correct.item() /
                                len(test_dataset)))
        test_mark(net)


def test_mark(net):
    labels = [1, 5, 0, 0, 4, 5, 2, 3, 1, 2, 7, 7, 1, 1, 1, 3, 2, 6, 2, 3, 7, 7, 1, 7, 7, 6, 2, 7, 2, 1, 7, 4, 1, 2, 0,
              6, 2, 0, 5, 3, 7, 0, 4, 3, 2, 6, 4, 6, 5, 6, 6, 4, 1, 6, 4, 3, 6, 4, 4, 1, 3, 7, 3, 7, 0, 5, 4, 4, 4, 7,
              1, 4, 6, 1, 4, 3, 5, 6, 1, 5, 1, 0, 5, 1, 3, 3, 0, 4, 2, 6, 1, 0, 6, 5, 6, 3, 4, 4, 2, 3, 7, 2, 0, 7, 4,
              5, 1, 7, 7, 1, 5, 3, 1, 4, 7, 7, 1, 6, 7, 6, 0, 6, 0, 2, 5, 6, 5, 4, 3, 5, 2, 1, 0, 4, 3, 5, 2, 3, 3, 3,
              4, 5, 0, 1, 6, 5, 5, 7, 2, 1, 1, 3, 0, 3, 5, 6, 4, 1, 3, 7, 1, 6, 4, 0, 2, 4, 5, 2, 6, 6, 5, 6, 5, 0, 7,
              0, 5, 7, 6, 6, 2, 7, 3, 5, 0, 2, 7, 4, 1, 1, 2, 0, 6, 3, 1, 3, 3, 5, 4, 4, 3, 6, 5, 2, 0, 0, 4, 1, 4, 0,
              3, 0, 2, 3, 0, 3, 2, 2, 3, 2, 5, 4, 0, 2, 5, 0, 2, 1, 6, 4, 0, 6, 3, 0, 5, 0, 3, 3, 0, 2, 4, 3, 0, 6, 3,
              6, 3, 0, 0, 0, 0, 5, 1, 2, 7, 6, 0, 4, 1, 5, 6, 1, 2, 5, 5, 0, 6, 2, 3, 6, 3, 1, 6, 0, 1, 6, 1, 4, 6, 0,
              0, 7, 2, 5, 0, 1, 4, 5, 5, 7, 5, 2, 4, 0, 1, 4, 5, 2, 6, 3, 0, 4, 6, 1, 2, 3, 7, 2, 1, 5, 4, 7, 3, 3, 5,
              2, 3, 0, 6, 3, 3, 2, 5, 6, 4, 1, 5, 6, 3, 7, 4, 5, 3, 7, 4, 5, 2, 5, 6, 6, 3, 4]

    net.eval()  # 将模型变换为测试模式

    pic = "0_7480.png"
    img = cv2.imread(os.path.join('./pic/train', pic), cv2.IMREAD_GRAYSCALE)
    res = []
    length = len(labels)
    labels = labels * 10
    begin = 0
    tmp = img.copy()
    for k in range(20):
        tmp = np.append(tmp, [[labels[(begin + k)] / 8 * 255 for p in range(28)]], axis=0)
    cv2.imwrite("test.png", tmp)
    for i in range(1,length*8+1):

        tmp = np.resize(tmp, (1, 1, 48, 28))
        tmp=tmp/255
        input=torch.from_numpy(tmp)
        a,pre=net(input)
        a=(torch.argmax(a, dim=1))
        a=a.numpy()[0]
        res.append(a)
        tmp=img.copy()
        for k in range(19):
            tmp = np.append(tmp, [[labels[(i + k)] / 8 * 255 for p in range(28)]], axis=0)
        tmp = np.append(tmp, [[a / 8 * 255 for p in range(28)]], axis=0)
        #print(pre)
    ans=[]
    for i in range(length):
        map={}
        for j in range(10):
            map[j]=0
        for j in range(i,len(res),length):
            map[res[j]]=map[res[j]]+1
        maxnum=-1
        index=0
        for key, value in map.items():
            #print(key,value)
            if value>maxnum:
                maxnum=value
                index=key
        ans.append(index)
    sum = 0
    for i, val in enumerate(res):
        if val == labels[(i + 20) % len(labels)]:
            sum = sum + 1
    print(sum / len(res))
    return sum / len(res)

if __name__ == '__main__':
    pune(0.01)
