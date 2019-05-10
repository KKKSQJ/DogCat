#conding:utf-8
import os
import cv2
from config import opt
import torch as t
import models
from PIL import Image
from torchvision import transforms as T
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.image as mpimg

def display(images,labels,cols=5,norm=None, interpolation=None, cmap=None):
    rows = len(images) // cols + 1
    plt.figure(figsize=(12, 12 * rows // cols))
    i = 1
    for image,label in zip(images, labels):
        plt.subplot(rows, cols, i)
        plt.title(label, fontsize=9)
        # plt.imshow(image.astype(np.uint8), cmap=cmap,
        #            norm=norm, interpolation=interpolation)
        plt.imshow(image, cmap=cmap,
                   norm=norm, interpolation=interpolation)
        plt.axis('off')
        i += 1

    plt.show()

def write_csv(results,file_name):
    import csv
    with open(file_name,'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id','label'])
        writer.writerows(results)

dir = r'G:\project\data\dogcat\test1'

def detect(**kwargs):
    opt._parse(kwargs)
    images = []
    labels =[]

    # configure model
    model = getattr(models, opt.model)().eval()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    model.to(opt.device)

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])




    for ii, i in enumerate(np.random.choice(os.listdir(dir), 25)):


        img = Image.open(os.path.join(dir, i))
        input_img = T.Resize(224)(img)
        input_img = T.CenterCrop(224)(input_img)
        input_img = T.ToTensor()(input_img)
        input_img = normalize(input_img)
        input_img = input_img.unsqueeze(0)
        images.append(img)

        input = input_img.to(opt.device)
        score = model(input)
        probability = t.nn.functional.softmax(score, dim=1)[:, 0].detach().tolist()
        label = score.max(dim = 1)[1].detach().tolist()
        labels.append(label)
        print(label)
    display(images, labels)





if __name__ == '__main__':
    detect()


