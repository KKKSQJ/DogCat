#coding:utf8
import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T

class DogCat(data.dataset):
    def __init__(self, root, transforms=None, train=True, test=False):
        self.test = test
        imgs = [os.path.join(root, img) for img in os.listdir(root)]

        #test:data/test1/8973.jpg
        #train:data/train/cat.10004.jpg
        if self.test:
            imgs = sorted(imgs, key=lambda x:int(x.split('.')[-2].split('/')[-1]))
        else:
            imgs = sorted(imgs, key=lambda x:int(x.split('.')[-2]))

        imgs_len = len(imgs)

        if self.test:
            self.imgs = imgs
        elif self.train:
            self.imgs = imgs[:int(0.7*imgs_len)]
        else:
            self.imgs = imgs[int(0.7*imgs_len):]

        if transforms is None:
            normalize = T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])

            if self.test or not train:
                self.transforms = T.Compose(
                    [T.Scale(224),
                     T.CenterCrop(224),
                     T.ToTensor(),
                    ]
                )

            else:
                self.transforms = T.Compose([
                    T.Scale(256),
                    T.RandomSizedCrop(224),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize
                    ])

    def __getitem__(self, index):
        img_path = self.imgs[index]
        if self.test: label = int(self.imgs[index].split('.')[-2].split('/')[-1])
        else: label = 1 if 'dog' in img_path.split('/')[-1] else 0
        data = Image.open(img_path)
        data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)
