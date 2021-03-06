import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
import cv2
# Press the green button in the gutter to run the script.


train_dataset = datasets.MNIST(root='./num/',
                                   train=True,
                                   transform=transforms.ToTensor(),
                                   download=False)
test_dataset = datasets.MNIST(root='./num/',
                                  train=False,
                                  transform=transforms.ToTensor,
                                  download=False)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=64,
                                               shuffle=True)
    # 装载测试集
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=64,
                                              shuffle=True)
idx=0


for i, data in enumerate(test_loader):
    inputs, labels = data

    pics=inputs.numpy()
    label=labels.numpy()


    for j,pic in enumerate(pics):
        #print("./pic/train/%d_%d.png"%(label[j],idx))
        cv2.imwrite("./pic/test/%d_%d.png"%(label[j],idx),pic[0]*255)

        idx=idx+1
