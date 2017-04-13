import argparse
import numpy as np
import os
import shutil

parser = argparse.ArgumentParser()

parser.add_argument('path', metavar='train/0', help='the data path')
parser.add_argument('num0', metavar='M', type=int, help='the number of degree 0 data')
parser.add_argument('num', metavar='N', type=int, help='the number of current degree data')

args = parser.parse_args()

ii = int(args.num0/args.num)

from glob import glob

fs = glob('{}/*.jpeg'.format(args.path))

files = np.array(sorted(fs))

for i in range(ii):
    for file in files:
        basename = os.path.basename(file).split('.')[0]
        newname = args.path + '/' + basename+'_{}.jpeg'.format(i)
        shutil.copy(file, newname)
