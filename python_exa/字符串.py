# -*- coding: utf-8 -*-

# @Time    : 2019-03-28 10:16
# @Author  : jian
# @File    : 字符串.py
import os
x = "abc"
y = "hjk"
z = ["h", "k"]
#
# m = x.join(y)
# print(m)

# m = x.join(z)
# print(m)

import dis


def func(a, b):
    a, b = b, a
    print(a, b)


a = 10
b = 20
func(a, b)
dis.dis(func)