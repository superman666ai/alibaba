# -*- coding: utf-8 -*-

# @Time    : 2019-03-28 13:48
# @Author  : jian
# @File    : magic_method.py

class FunctionalList(object):
    ''' 实现了内置类型list的功能,并丰富了一些其他方法: head, tail, init, last, drop, take'''

    def __init__(self, values=None):
        if values is None:
            self.values = []
        else:
            self.values = values

    def __len__(self):
        return len(self.values)

    def __getitem__(self, key):
        return self.values[key]

    def __setitem__(self, key, value):
        self.values[key] = value

    def __delitem__(self, key):
        del self.values[key]

    def __iter__(self):
        return iter(self.values)

    def __reversed__(self):
        return FunctionalList(reversed(self.values))

    def append(self, value):
        self.values.append(value)

    def head(self):
        # 获取第一个元素
        return self.values[0]

    def tail(self):
        # 获取第一个元素之后的所有元素
        return self.values[1:]

    def init(self):
        # 获取最后一个元素之前的所有元素
        return self.values[:-1]

    def last(self):
        # 获取最后一个元素
        return self.values[-1]

    def drop(self, n):
        # 获取所有元素，除了前N个
        return self.values[n:]

    def take(self, n):
        # 获取前N个元素
        return self.values[:n]


from collections import Counter, OrderedDict

tup = [[1, "abc"], [2, "def"]]

dic = OrderedDict(tup)

# dic["name"] = "xiaoming"
# dic["age"] = 20

print(dic)

"""
__call__ 
允许一个类的实例像函数一样被调用。
实质上说，这意味着 x() 与 x.__call__() 是相同的。
注意 __call__ 的参数可变。这意味着你可以定义 __call__ 为其他你想要的函数，无论有多少个参数。
"""


class Entity:
    """
    调用实体来改变实体的位置
    """

    def __init__(self, size, x, y):
        self.x, self.y = x, y
        self.size = size

    def __call__(self, x, y):
        """
        改变实体的位置
        """
        self.x, self.y = x, y

# a = Entity(100, 10, 20)
# a("x", "d")
# print(a.x, a.y)