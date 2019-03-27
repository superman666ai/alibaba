# -*- coding: utf-8 -*-

# @Time    : 2019-03-27 16:26
# @Author  : jian
# @File    : 列表.py

lis = [1, 2, 2, 3, 4, 54, 43, 4321, 3, 2, 1]
# 去重
lis = list(set(lis))

print(lis)

def fun(x):
    return x**2
lis = map(fun, lis)
# for i in lis:
#     print(i)
lis = [i for i in lis if i > 10]

print(lis)