# -*- coding: utf-8 -*-

# @Time    : 2019/1/10 17:42
# @Author  : jian
# @File    : read_data.py
import pandas as pd
import seaborn
import matplotlib.pyplot as plt

# 读取数据 分析
train = pd.read_csv('data/zhengqi_train.txt', sep='\t')
test = pd.read_csv('data/zhengqi_test.txt', sep='\t')
train_x = train.drop(['target'], axis=1)


# # 画图分析
# all_data = pd.concat([train_x, test])
# name = 0
# for col in all_data.columns:
#     seaborn.distplot(train[col])
#     seaborn.distplot(test[col])
#     # plt.show()
#     plt.savefig("image/" + "V" + str(name) + ".png")
#     name += 1
#     plt.close()
#

# 删除特征因子
# all_data.drop(['V5', 'V17', 'V28', 'V22', 'V11', 'V9'], axis=1, inplace=True)


