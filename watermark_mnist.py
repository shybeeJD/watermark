import torch
import torch.nn as nn
import os
import cv2
from lstm import GRUNet
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch.optim as optim
import random
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
        self.gruNet.load_state_dict(torch.load("models/gru_5000_noisy.pth"))
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
    # ???????????????????????????
    criterion = nn.CrossEntropyLoss()
    # ?????????????????? Adam ?????????????????????
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

            optimizer.zero_grad()  # ???????????????
            outputs,mark = net(inputs)  # ???????????????????????????????????????

            loss = criterion(outputs, labels)  # ??????????????????
            loss.backward()  # ????????????
            optimizer.step()  # ?????????????????????????????????

            # print(loss)
            sum_loss += loss.item()
            if i % 100 == 99:
                print('[%d,%d] loss:%.03f' %
                      (epoch + 1, i + 1, sum_loss / 100))
                sum_loss = 0.0
        net.eval()  # ??????????????????????????????
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
def test():
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
    net.eval()  # ??????????????????????????????
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

def test_mark(labels):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = WaterMarkedNet().to(device)
    net = net.double()
    net.load_state_dict(torch.load("models/MNIST_1000.pth"))
    net.eval()  # ??????????????????????????????
    pic="0_7480.png"
    img = cv2.imread(os.path.join('./pic/train', pic), cv2.IMREAD_GRAYSCALE)
    res=[]
    length = len(labels)
    labels=labels*6
    begin=0
    tmp=img.copy()
    for k in range(20):
        tmp = np.append(tmp, [[labels[(begin + k )] / 8 * 255 for p in range(28)]], axis=0)
    cv2.imwrite("test.png",tmp)

    for i in range(1,length*1):
        tmp = np.resize(tmp, (1, 1, 48, 28))
        tmp=tmp/255
        input=torch.from_numpy(tmp)
        a,pre=net(input)
        pre=(torch.argmax(pre, dim=1))
        pre=pre.numpy()[0]
        res.append(pre)
        tmp=img.copy()
        for k in range(19):
            tmp = np.append(tmp, [[labels[(i + k)] / 8 * 255 for p in range(28)]], axis=0)
        tmp = np.append(tmp, [[pre / 8 * 255 for p in range(28)]], axis=0)
        #print(pre)
    return res
def test_mark_noisy(labels):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = WaterMarkedNet().to(device)
    net = net.double()
    net.load_state_dict(torch.load("models/MNIST_1000.pth"))
    net.eval()  # ??????????????????????????????
    pic="0_7480.png"
    img = cv2.imread(os.path.join('./pic/train', pic), cv2.IMREAD_GRAYSCALE)
    res=[]
    length = len(labels)
    labels=labels*6
    begin=0
    tmp=img.copy()
    for k in range(20):
        tmp = np.append(tmp, [[random.randint(0,7) / 8 * 255 for p in range(28)]], axis=0)
    cv2.imwrite("test.png",tmp)

    for i in range(1,length*1):
        tmp = np.resize(tmp, (1, 1, 48, 28))
        tmp=tmp/255
        input=torch.from_numpy(tmp)
        a,pre=net(input)
        pre=(torch.argmax(pre, dim=1))
        pre=pre.numpy()[0]
        res.append(pre)
        tmp=img.copy()
        for k in range(19):
            tmp = np.append(tmp, [[random.randint(0,7) / 8 * 255 for p in range(28)]], axis=0)
        tmp = np.append(tmp, [[pre / 8 * 255 for p in range(28)]], axis=0)
        #print(pre)
    return res

def test_sign():
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
    res=test_mark_noisy(labels)
    sum=len(labels)-20
    acc=0
    for i in range(20,len(labels)):
        if labels[i]==res[i-20]:
            acc=acc+1

    print(acc/sum)
if __name__ == '__main__':
    test()