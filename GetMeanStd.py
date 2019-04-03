import numpy as np
import cv2
import random
from tqdm import tqdm
import torch

train_txt_path = 'train.txt'
Max_num = 10000
img_h, img_w = 32, 32
imgs = np.zeros([img_w, img_h, 3, 1])
means, stdevs = [], []

with open(train_txt_path,'r') as f:
    lines = f.readlines()
    random.shuffle(lines)
    for i in tqdm(range(Max_num)):
        img_path = lines[i].strip()
        img = cv2.imread(img_path)
        img = cv2.resize(img, (img_w, img_h), interpolation=cv2.INTER_CUBIC)
        img = img[:, :, :, np.newaxis]
        imgs = np.concatenate((imgs, img), axis=3)

imgs = imgs.astype(np.float32)/255.

for i in range(3):
    pixels = imgs[:, :, i, :].ravel()
    means.append(np.mean(pixels))
    stdevs.append(np.std(pixels))

means.reverse()
stdevs.reverse()

print("normMean = {}".format(means))
print("normStd = {}".format(stdevs))
print('transforms.Normalize(normMean = {}, normStd = {})'.format(means, stdevs))

