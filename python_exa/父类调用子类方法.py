# -*- coding: utf-8 -*-

# @Time    : 2019-04-02 15:00
# @Author  : jian
# @File    : 父类调用子类方法.py
from operator import methodcaller


class B():
    pass


class A(B):
    pass


class C(B):
    def pr(self):
        print("11111")


a = A()

print(hasattr(a, "pr"))

print(A.__bases__)

# 获取继承列表

print(C.mro())


def findbases(kls, topclass):
    retval = list()
    for base in kls.__bases__:
        if issubclass(base, topclass):
            retval.extend(findbases(base, topclass))
            retval.append(base)
    return retval
