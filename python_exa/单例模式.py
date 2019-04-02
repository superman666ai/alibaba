# -*- coding: utf-8 -*-

# @Time    : 2019-03-19 14:08
# @Author  : jian
# @File    : 单例模式.py
"""
要想弄明白为什么每个对象被实例化出来之后,
都会重新被分配出一块新的内存地址, 就要清楚一个python中的内置函数__new__(),
它跟__init__()一样, 都是对象在被创建出来的时候, 就自动执行的一个函数,
init()函数,是为了给函数初始化属性值的
而__new__()这个函数, 就是为了给对象在被实例化的时候,
分配一块内存地址, 因此, 我们可以重写__new__()这个方法
让他在第一次实例化一个对象之后, 分配一块地址, 在此后的所有实例化的其他对象时,
都不再分配新的地址, 而继续使用第一个对象所被分配的地址,
"""


# 第一种方法 使用 __new__  推荐使用 方便快捷
class B():
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
            cls._instance = super(B, cls).__new__(cls, *args, **kwargs)
        return cls._instance


d = B()
print('d对象所在的内存地址是 %d, B类所在的内存地址是 %d' % (id(d), id(B())))
e = B()
print('e对象所在的内存地址是 %d, B类所在的内存地址是 %d' % (id(e), id(B())))
f = B()
print('f对象所在的内存地址是 %d, B类所在的内存地址是 %d' % (id(f), id(B())))

# 第二种方法 使用 类 @classmethod  并行执行的时候会出现问题 加锁
# 这种方式实现的单例模式，使用时会有限制，以后实例化必须通过 obj = Singleton.instance()

# import time
# import threading
# class Singleton(object):
#     _instance_lock = threading.Lock()
#
#     def __init__(self):
#         time.sleep(1)
#
#     @classmethod
#     def instance(cls, *args, **kwargs):
#         if not hasattr(Singleton, "_instance"):
#             with Singleton._instance_lock:
#                 if not hasattr(Singleton, "_instance"):
#                     Singleton._instance = Singleton(*args, **kwargs)
#         return Singleton._instance

# 第三种方法 基于metaclass 方式实现

import threading


class SingletonType(type):
    _instance_lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
            with SingletonType._instance_lock:
                if not hasattr(cls, "_instance"):
                    cls._instance = super(SingletonType, cls).__call__(*args, **kwargs)
        return cls._instance


class Foo(metaclass=SingletonType):
    def __init__(self):
        self.name = False


d = Foo()
print('d对象所在的内存地址是 %d, B类所在的内存地址是 %d' % (id(d), id(Foo)))
e = Foo()
print('e对象所在的内存地址是 %d, B类所在的内存地址是 %d' % (id(e), id(Foo)))
f = Foo()
print('f对象所在的内存地址是 %d, B类所在的内存地址是 %d' % (id(f), id(Foo)))


# 创建实例时把所有实例的__dict__指向同一个字典,这样它们都具有相同的属性和方法(类的__dict__存储对象属性)
class Singleton(object):
    _state = {}
    def __new__(cls, *args, **kwargs):
        ob = super(Singleton,cls).__new__(cls, *args, **kwargs)
        ob.__dict__ = cls._state
        return ob

# 类B即为单例类
class A(Singleton):
    pass


dd = A()
print('d对象所在的内存地址是 %d, B类所在的内存地址是 %d' % (id(dd), id(A)))
ee = A()
print('e对象所在的内存地址是 %d, B类所在的内存地址是 %d' % (id(ee), id(A)))
ff = A()
print('f对象所在的内存地址是 %d, B类所在的内存地址是 %d' % (id(ff), id(A)))
