# -*- coding: utf-8 -*-

# @Time    : 2019-03-28 9:42
# @Author  : jian
# @File    : 基本函数.py

"""
map() 会根据提供的函数对指定序列做映射。
第一个参数 function 以参数序列中的每一个元素调用 function 函数，返回包含每次 function 函数返回值的新列表。
"""

lis = [1, 2, 2, 3, 4, 54, 43, 4321, 3, 2, 1]
# 去重
lis = list(set(lis))
# 返回列表的平方
def fun(x):
    return x**2
lis = map(fun, lis)
print(lis)

"""
filter() 函数用于过滤序列，过滤掉不符合条件的元素，返回由符合条件元素组成的新列表。
该接收两个参数，第一个为函数，第二个为序列，序列的每个元素作为参数传递给函数进行判，
然后返回 True 或 False，最后将返回 True 的元素放到新列表中。
"""
# 是否为奇数
def is_odd(n):
    return n % 2 == 1

lis = filter(is_odd, lis)
print(list(lis))