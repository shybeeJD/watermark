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
label_list=[1, 5, 0, 0, 4, 5, 2, 3, 1, 2, 7, 7, 1, 1, 1, 3, 2, 6, 2, 3, 7, 7, 1, 7, 7, 6, 2, 7, 2, 1, 7, 4, 1, 2, 0, 6, 2, 0, 5, 3, 7, 0, 4, 3, 2, 6, 4, 6, 5, 6, 6, 4, 1, 6, 4, 3, 6, 4, 4, 1, 3, 7, 3, 7, 0, 5, 4, 4, 4, 7, 1, 4, 6, 1, 4, 3, 5, 6, 1, 5, 1, 0, 5, 1, 3, 3, 0, 4, 2, 6, 1, 0, 6, 5, 6, 3, 4, 4, 2, 3, 7, 2, 0, 7, 4, 5, 1, 7, 7, 1, 5, 3, 1, 4, 7, 7, 1, 6, 7, 6, 0, 6, 0, 2, 5, 6, 5, 4, 3, 5, 2, 1, 0, 4, 3, 5, 2, 3, 3, 3, 4, 5, 0, 1, 6, 5, 5, 7, 2, 1, 1, 3, 0, 3, 5, 6, 4, 1, 3, 7, 1, 6, 4, 0, 2, 4, 5, 2, 6, 6, 5, 6, 5, 0, 7, 0, 5, 7, 6, 6, 2, 7, 3, 5, 0, 2, 7, 4, 1, 1, 2, 0, 6, 3, 1, 3, 3, 5, 4, 4, 3, 6, 5, 2, 0, 0, 4, 1, 4, 0, 3, 0, 2, 3, 0, 3, 2, 2, 3, 2, 5, 4, 0, 2, 5, 0, 2, 1, 6, 4, 0, 6, 3, 0, 5, 0, 3, 3, 0, 2, 4, 3, 0, 6, 3, 6, 3, 0, 0, 0, 0, 5, 1, 2, 7, 6, 0, 4, 1, 5, 6, 1, 2, 5, 5, 0, 6, 2, 3, 6, 3, 1, 6, 0, 1, 6, 1, 4, 6, 0, 0, 7, 2, 5, 0, 1, 4, 5, 5, 7, 5, 2, 4, 0, 1, 4, 5, 2, 6, 3, 0, 4, 6, 1, 2, 3, 7, 2, 1, 5, 4, 7, 3, 3, 5, 2, 3, 0, 6, 3, 3, 2, 5, 6, 4, 1, 5, 6, 3, 7, 4, 5, 3, 7, 4, 5, 2, 5, 6, 6, 3, 4]

class Lstm(nn.Module):
    def __init__(self,input_size=1):
        super(Lstm, self).__init__()
        self.lstm=nn.LSTM(input_size, 30, 2)

        self.fc1 = nn.Sequential(
            nn.Linear(10, 30),
            nn.BatchNorm1d(30),
            nn.ReLU(),
            nn.Linear(20, 1))



    def forward(self, x):
        x=self.lstm(x)
        x=self.fc1(x)
        return x
def dataset_gen(label_list, paddings, nums, bits, replace):
    length = len(label_list)
    labels = label_list * 2
    data = []
    label= []
    for i in range(length):
        data.append(labels[i:i + paddings])
        label.append(labels[i + paddings])
        for j in range(nums):
            for k in range(replace):
                tmp=labels[i:i + paddings].copy()
                idx=random.randint(0,paddings-1)
                value=random.randint(0,2**bits-1)
                data.append(tmp)
                data[-1][idx]=value
                #print(data[-1])
                label.append(labels[i + paddings])
    return data,label
data,label=dataset_gen(label_list,5,100,3,1)
print(len(data))
data=np.array(data)
print(data.shape)

