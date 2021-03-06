import os
from torch.utils.data import DataLoader
import cv2
from torchvision import datasets, transforms
import numpy as np
import random
import torch
label_list=[1, 5, 0, 0, 4, 5, 2, 3, 1, 2, 7, 7, 1, 1, 1, 3, 2, 6, 2, 3, 7, 7, 1, 7, 7, 6, 2, 7, 2, 1, 7, 4, 1, 2, 0, 6, 2, 0, 5, 3, 7, 0, 4, 3, 2, 6, 4, 6, 5, 6, 6, 4, 1, 6, 4, 3, 6, 4, 4, 1, 3, 7, 3, 7, 0, 5, 4, 4, 4, 7, 1, 4, 6, 1, 4, 3, 5, 6, 1, 5, 1, 0, 5, 1, 3, 3, 0, 4, 2, 6, 1, 0, 6, 5, 6, 3, 4, 4, 2, 3, 7, 2, 0, 7, 4, 5, 1, 7, 7, 1, 5, 3, 1, 4, 7, 7, 1, 6, 7, 6, 0, 6, 0, 2, 5, 6, 5, 4, 3, 5, 2, 1, 0, 4, 3, 5, 2, 3, 3, 3, 4, 5, 0, 1, 6, 5, 5, 7, 2, 1, 1, 3, 0, 3, 5, 6, 4, 1, 3, 7, 1, 6, 4, 0, 2, 4, 5, 2, 6, 6, 5, 6, 5, 0, 7, 0, 5, 7, 6, 6, 2, 7, 3, 5, 0, 2, 7, 4, 1, 1, 2, 0, 6, 3, 1, 3, 3, 5, 4, 4, 3, 6, 5, 2, 0, 0, 4, 1, 4, 0, 3, 0, 2, 3, 0, 3, 2, 2, 3, 2, 5, 4, 0, 2, 5, 0, 2, 1, 6, 4, 0, 6, 3, 0, 5, 0, 3, 3, 0, 2, 4, 3, 0, 6, 3, 6, 3, 0, 0, 0, 0, 5, 1, 2, 7, 6, 0, 4, 1, 5, 6, 1, 2, 5, 5, 0, 6, 2, 3, 6, 3, 1, 6, 0, 1, 6, 1, 4, 6, 0, 0, 7, 2, 5, 0, 1, 4, 5, 5, 7, 5, 2, 4, 0, 1, 4, 5, 2, 6, 3, 0, 4, 6, 1, 2, 3, 7, 2, 1, 5, 4, 7, 3, 3, 5, 2, 3, 0, 6, 3, 3, 2, 5, 6, 4, 1, 5, 6, 3, 7, 4, 5, 3, 7, 4, 5, 2, 5, 6, 6, 3, 4]
print(len(label_list))

train_dataset = datasets.MNIST(root='./num/',
                                   train=True,
                                   transform=transforms.ToTensor(),
                                   download=False)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=64,
                                               shuffle=True)
test_dataset = datasets.MNIST(root='./num/',
                                  train=False,
                                  transform=transforms.ToTensor,
                                  download=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=64,
                                              shuffle=True)

def ensure_watermark(labels,paddings):
    length=len(labels)
    labels=labels*2
    print(len(labels))
    map=[]
    for i in range(length):
        map.append([''.join("%s"%a for a in labels[i:i+paddings]),labels[i+paddings]])
    return map

def marked_pic_gen(map,paddings,bits,times,isnormal=False):
    idx=0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        pics = inputs.numpy()
        label = labels.numpy()

        map_length=len(map)


        for j, pic in enumerate(pics):
            d,r,c=pic.shape
            b=np.zeros((1,paddings,c))
            randi=random.randint(0,map_length-1)
            if not isnormal:
                for row in range(paddings):
                    for col in range(c):
                        b[0][row][col]=int(map[randi][0][row])/(2**bits)
            pic=np.append(pic,b,axis=1)

            # print("./pic/train/%d_%d.png"%(label[j],idx))
            if isnormal:
                cv2.imwrite("./marked_pic/train/%d_%d_%d.png" % (label[j], idx, times), pic[0] * 255)
            else:
                cv2.imwrite("./marked_pic/train/%d_%d_%d.png" % (map[randi][1], idx, times), pic[0] * 255)
            print(idx)
            idx = idx + 1
a=np.ones((5,6))
print(a)
marked_pic_gen(ensure_watermark(label_list,5),5,3,1,True)