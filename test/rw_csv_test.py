# import argparse
#
# parser = argparse.ArgumentParser()
#
# parser.add_argument('-data', default='./trainLabels.csv', metavar='./trainLabels.csv', help='input label data')
#
# args = parser.parse_args()
#
# print('Input csv file: {}'.format(args.data))
#
#
# import pandas as pd
#
# labels = pd.read_csv('./trainLabels.csv')
#
# print(labels.iloc[0])
#
# imagelist = ['0_left.jpeg', '1_left.jpeg', '2_left.jpeg', '3_left.jpeg',
#              '4_left.jpeg', '5_left.jpeg', '6_left.jpeg', '7_left.jpeg', '8_left.jpeg']
# catalist = []
#
# for i in imagelist:
#     if '0' in i:
#         catalist.append(True)
#     else:
#         catalist.append(False)
#
# print(catalist)
#
# list = ['image', 'level']
# cols = pd.DataFrame(columns=list)
#
# datas = {}
# datas['image'] = imagelist
# datas['level'] = catalist
#
# for id in list:
#     cols[id] = datas[id]
#
# cols.to_csv('test.csv', index=False)


import pandas as pd

imagelist = ['0_left.jpeg', '1_left.jpeg', '2_left.jpeg', '3_left.jpeg',
             '4_left.jpeg', '5_left.jpeg', '6_left.jpeg', '7_left.jpeg', '8_left.jpeg']
catalist = []

for i in imagelist:
    if '0' in i:
        catalist.append(0)
    else:
        catalist.append(1)

list = ['image', 'level']
cols = pd.DataFrame(columns=list)


datas = {}
datas['image'] = imagelist
datas['level'] = catalist

for id in list:
    cols[id] = datas[id]

cols.to_csv('test.csv', index=False)
