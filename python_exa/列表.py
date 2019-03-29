# -*- coding: utf-8 -*-

# @Time    : 2019-03-27 16:26
# @Author  : jian
# @File    : 列表.py

lis = [1, 2, 2, 3, 4, 54, 43, 4321, 3, 2, 1]
# 去重
lis = list(set(lis))


# print(lis)

def fun(x):
    return x ** 2


lis = map(fun, lis)
# for i in lis:
#     print(i)
lis = [i for i in lis if i > 10]

# print(lis)

str = "fashdfuah3gyudsfuisahf"

lis = list(set(str))
lis.sort()
str = "".join(lis)
# print(str)

# 展开列表
"""
variable = [out_exp_res for out_exp in input_list if out_exp == 2]
  out_exp_res:　　列表生成元素表达式，可以是有返回值的函数。
  for out_exp in input_list：　　迭代input_list将out_exp传入out_exp_res表达式中。
  if out_exp == 2：　　根据条件过滤哪些值可以。
"""
a = [[1, 2], [3, 4], [5, 6]]

b = [j for i in a for j in i]

# print(b)

x = [j for i in a for j in i]

# print(x)

import numpy as np

c = np.array(a).flatten().tolist()
# print(c)

