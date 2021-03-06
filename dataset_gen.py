import cv2
import os
from sign_check import get_labels
import numpy as np


def dateset_gen(paddings):
    files= os.listdir('./pic/train')
    l=len(files)

    labels = get_labels("123", 10, 5)
    sign_len=len(labels)
    print(labels)

    begin=0
    for i in range(1):
        rand_idx = np.random.randint(0, l, 500)
        for j in rand_idx:
            file=files[j]
            if file.endswith(".png"):

                img=cv2.imread(os.path.join('./pic/train',file),cv2.IMREAD_GRAYSCALE)
                for k in range(paddings):
                    img=np.append(img,[[labels[(begin+k+sign_len)%sign_len]/10*255 for p in range(28)]],axis=0)
                print(labels[(begin+0+sign_len)%sign_len],end=", ")
                cv2.imwrite("./pic/trigger/%s_%s_new_%d.png"%(labels[(begin+paddings+sign_len)%sign_len],file.split('_')[0],i*len(rand_idx)+j),img)
                begin=(begin+1+sign_len)%sign_len





dateset_gen(5)
