# # read file
#
# training_loss = []
#
# with open('RMSprop0.log') as file:
#     lines = file.readlines()
#     for line in lines:
#         if line.find('[60/61]') != -1:
#             words = line.split(' ')
#             num = float(words[7].split('(')[1].split(')')[0])
#             training_loss.append(num)
#
# print(training_loss)
#
#
# # plot
#
# import matplotlib.pyplot as plt
#
# plt.figure(0)
# plt.subplot(324)
# plt.plot(training_loss)
# plt.show()

import argparse
import math
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument('-l','--list', nargs='+', help='<Required> Set flag', required=True)

args = parser.parse_args()

listsize = len(args.list)

row = int(math.ceil(listsize/2.))


def plot_training_loss(logpath,row, column, index):
    training_loss = []
    with open(logpath) as file:
        lines = file.readlines()
        for line in lines:
            if line.find('[60/61]') != -1:
                words = line.split(' ')
                num = float(words[7].split('(')[1].split(')')[0])
                training_loss.append(100*num)
    grid = str((row*100+column*10+index))
    plt.subplot(grid)
    plt.ylim((70,80))
    print(training_loss)
    plt.xlabel(logpath.split('.')[0])
    plt.plot(training_loss)

plt.figure('training loss')
i = 0
for l in args.list:
    plot_training_loss(l, row, 2, i)
    i = i+1
plt.show()

