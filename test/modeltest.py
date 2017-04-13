# 1. load model
import models

from PIL import Image

import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import os

import pandas as pd


def main():
    model = models.ResNet2()
    checkpoint = torch.load('/Users/zhangweidong03/Code/dl/pytorch/examples/imagenet/model_best.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_loader = torch.utils.data.DataLoader(
        models.ImageFolder1('../sample', transforms.Compose([
            transforms.Scale(512),
            transforms.CenterCrop(448),
            transforms.ToTensor(),
            normalize,
        ])),batch_size=4, shuffle=False)


    trans = transforms.Compose([
        transforms.Scale(512),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        normalize,
    ])

    model.eval()


    reslist = []
    nameslist = []

    for i, (input, target, path) in enumerate(val_loader):
        input_var = torch.autograd.Variable(input)
        output = model(input_var)
        res1,pred = output.topk(1, 1, True, True)
        outlist = pred.data.numpy()
        for o in outlist:
            reslist.append(o[0])
        for p in path:
            nameslist.append(p)

    print('result list: {}'.format(reslist))
    print('names list: {}'.format(nameslist))

    datas = {}
    datas['image'] = nameslist
    datas['level'] = reslist

    list = ['image', 'level']
    cols = pd.DataFrame(columns=list)

    for id in list:
        cols[id] = datas[id]

    cols.to_csv('test.csv', index=False)

    # img = Image.open('../sample/4/17_left.jpeg')
    #
    # in_img = trans(img)
    #
    # in_imgs = [ in_img for i in range(5)]
    #
    # in_imgs = torch.stack(in_imgs, 0)
    #
    # input_var = torch.autograd.Variable(in_imgs)
    #
    # model.eval()
    #
    # output = model(input_var)
    #
    # _, pred = output.topk(1,1,True,True)
    #
    # print(output)
    # print(pred)

if __name__ == '__main__':
    main()