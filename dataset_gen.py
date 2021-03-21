import cv2
import os
from sign_check import get_labels
import numpy as np
import random

def dateset_gen(paddings):
    files= os.listdir('./pic/train')
    l=len(files)

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
    sign_len=len(labels)

    begin=0
    for i in range(10):
        rand_idx = np.random.randint(0, l, 5000)
        for j in rand_idx:
            file=files[j]
            if file.endswith(".png"):

                img=cv2.imread(os.path.join('./pic/train',file),cv2.IMREAD_GRAYSCALE)
                for k in range(paddings):
                    img=np.append(img,[[labels[(begin+k+sign_len)%sign_len]/8*255 for p in range(28)]],axis=0)
                #print(labels[(begin+0+sign_len)%sign_len],end=", ")
                cv2.imwrite("./pic/trigger/%s_%s_new_%d.png"%(labels[(begin+paddings+sign_len)%sign_len],file.split('_')[0],i*len(rand_idx)+j),img)
                begin=(begin+1+sign_len)%sign_len

def noisy_gen(paddings):
    files = os.listdir('./pic/train')
    l = len(files)

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
    sign_len = len(labels)

    begin = 0
    for i in range(1):
        rand_idx = np.random.randint(0, l, 1000)
        for j in rand_idx:
            file = files[j]
            if file.endswith(".png"):

                img = cv2.imread(os.path.join('./pic/train', file), cv2.IMREAD_GRAYSCALE)
                for k in range(paddings):
                    img = np.append(img, [[random.randint(0,9) / 8 * 255 for p in range(28)]],
                                    axis=0)
                # print(labels[(begin+0+sign_len)%sign_len],end=", ")
                cv2.imwrite("./pic/trigger_test/%s_%s_new_%d.png" % (
                labels[(begin + paddings + sign_len) % sign_len], file.split('_')[0], i * len(rand_idx) + j), img)
                begin = (begin + 1 + sign_len) % sign_len


noisy_gen(20)
