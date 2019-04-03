#coding:utf-8
import os
import torch as t
import models
from config import opt
from data.dataset import DogCat
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchnet import meter
from utils.visualize import Visualizer
from tqdm import tqdm


def test(**kwargs):
    opt.parse(kwargs)
    model = getattr(models, opt.model)().eval()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu:
        model.cuda()

    test_data = DogCat(opt.test_data_root, test=True)
    test_dataloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    results = []
    for ii, (data, path) in tqdm(enumerate(test_dataloader)):
        input = t.autograd.Variable(data, volatile=True)
        if opt.use_gpu:
            input = input.cuda()
        score = model(input)
        probability = t.nn.functional.softmax(score)[:, 0].data.tolist()
        # label = score.max(dim = 1)[1].data.tolist()

        batch_results = [(path_, probability_) for path_, probability_ in zip(path, probability)]

        results += batch_results
    write_csv(results, opt.result_file)

def write_csv(results,file_name):
    import csv
    with open(file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label'])
        writer.writerows(results)

def train(**kwargs):
    opt.parse(kwargs)
    vis = Visualizer(opt.env)

    # step1: configure model
    model = getattr(models, opt.model)()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu:
        model.cuda()

    #step2: get data
    train_data = DogCat(opt.train_data_root, train=True)
    val_data = DogCat(opt.train_data_root, train=False)
    train_dataloadet = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)

    #step3:criterion and optimizer
    criterion = t.nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = t.optim.Adam(model.parameters(), lr=lr, weight_decay=opt.weight_decay)

    #step4:meters 一个轻量的计算库
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)
    previous_loss = 1e100

    #train
    for epoch in range(opt.max_epoch):

        loss_meter.reset()
        confusion_matrix.reset()

        #tqdm:一个Python中进度条加载的工具 不影响实际程序 只是为了美观
        for ii, (data, label) in tqdm(enumerate(train_dataloadet)):

            #train model
            input = Variable(data)
            target = Variable(label)

            #device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
            #input = input.to(device)
            #target = target.to(device)

            if opt.use_gpu:
                input = input.cuda()
                target = target.cuda()

            optimizer.zero_grad()
            score = model(input)
            loss = criterion(score,target)
            loss.backward()
            optimizer.step()

            #meters updata and Visualize
            loss_meter.add(loss.data[0])
            confusion_matrix.add(score.data,target.data)

            if ii%opt.print_freq==opt.print_freq-1:
                vis.plot('loss',loss_meter.value()[0])

                if os.path.exists(opt.debug_file):
                    import ipdb  #调试
                    ipdb.set_trace()

        model.save()

        # validata and visualize
        val_cm, val_accuracy = val(model, val_dataloader)

        vis.plot('val_accuracy', val_accuracy)
        vis.log("epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}".format(
            epoch=epoch, loss=loss_meter.value()[0], val_cm=str(val_cm.value()), train_cm=str(confusion_matrix.value()),
            lr=lr))

        if loss_meter.value()[0] > previous_loss:
            lr = lr*opt.lr_decay
            for param_group in optimizer.param_group:
                param_group['lr'] = lr

        previous_loss = loss_meter.value()[0]

def val(model,dataloader):
    model.eval()
    confusion_matrix = meter.ConfusionMeter(2)
    for ii, data in tqdm(enumerate(dataloader)):
        input, label = data
        val_input = Variable(input, volatile=True)
        val_label = Variable(label.type(t.LongTensor), volatile=True)
        if opt.use_gpu:
            val_input.cuda()
            val_label.cuda()
        score = model(val_input)
        confusion_matrix.add(score.data.squeeze(),label.type(t.LongTensor))

    model.train()
    cm_value = confusion_matrix.value()
    accuracy = 100.*(cm_value[0][0] + cm_value[1][1])/(cm_value.sum())
    return confusion_matrix, accuracy



if __name__=='__main__':
    import fire
    fire.Fire()




