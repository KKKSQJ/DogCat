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

def write_csv(results,file_name):
    import csv
    with open(file_name,'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id','label'])
        writer.writerows(results)

dir = r'G:\project\data\dogcat\test1'

def detect(**kwargs):
    opt._parse(kwargs)

    # configure model
    model = getattr(models, opt.model)().eval()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    model.to(opt.device)

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])

    gs1 = gridspec.GridSpec(5, 5)
    gs1.update(wspace=0.01, hspace=0.02)  # set the spacing between axes.
    plt.figure(figsize=(12, 12))


    for ii, i in enumerate(np.random.choice(os.listdir(dir), 25)):
        ax1 = plt.subplot(gs1[ii])
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect('equal')

        img = Image.open(os.path.join(dir, i))
        input_img = T.Resize(224)(img)
        input_img = T.CenterCrop(224)(input_img)
        input_img = T.ToTensor()(input_img)
        input_img = normalize(input_img)
        input_img = input_img.unsqueeze(0)

        input = input_img.to(opt.device)
        score = model(input)
        probability = t.nn.functional.softmax(score, dim=1)[:, 0].detach().tolist()
        label = score.max(dim = 1)[1].detach().tolist()
        print(label)

        # ax1.text(ii*5,  ii*5+ 8, str(label), color='w', size=11, backgroundcolor="none")
        plt.subplot(5, 5, ii + 1)
        plt.imshow(img)
        plt.axis('off')
    plt.show()




if __name__ == '__main__':
    detect()


