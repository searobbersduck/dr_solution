# 1. load model
import models

import argparse

from PIL import Image

import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import os

import pandas as pd

import ml_metrics

parser = argparse.ArgumentParser()
parser.add_argument('test_dataset', metavar='/test', help='the path of test dataset')
parser.add_argument('model', metavar='dr model', help = 'the trained model')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')

parser.add_argument('-kappa-val', dest='kappa_val', action='store_true', help='whether to check with quadratic kappa')

args = parser.parse_args()

def main():
    model = models.ResNet2()
    model = torch.nn.DataParallel(model).cuda()

    checkpoint = torch.load(args.model)
    model.load_state_dict(checkpoint['state_dict'])

    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_loader = torch.utils.data.DataLoader(
        models.ImageFolder1(args.test_dataset, transforms.Compose([
            transforms.Scale(512),
            transforms.CenterCrop(448),
            transforms.ToTensor(),
            normalize,
        ])),batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    trans = transforms.Compose([
        transforms.Scale(512),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        normalize,
    ])

    model.eval()


    reslist = []
    nameslist = []
    targetlist = []

    for i, (input, target, path) in enumerate(val_loader):
        print('batch: {}'.format(i))
        for t in target:
            targetlist.append(t)
        input_var = torch.autograd.Variable(input)
        output = model(input_var)
        res1,pred = output.topk(1, 1, True, True)
        outlist = pred.cpu().data.numpy()
        for o in outlist:
            reslist.append(o[0])
        for p in path:
            nameslist.append(p)

    print('result list: {}'.format(reslist))
    print('names list: {}'.format(nameslist))
    print('target list: {}'.format(targetlist))


    if args.kappa_val:
        kp = ml_metrics.quadratic_weighted_kappa(targetlist,reslist,0,4)
        print('quadratic weighted kappa: {}'.format(kp))


    datas = {}
    datas['image'] = nameslist
    datas['level'] = reslist

    list = ['image', 'level']
    cols = pd.DataFrame(columns=list)

    for id in list:
        cols[id] = datas[id]

    cols.to_csv('test.csv', index=False)

if __name__ == '__main__':
    main()