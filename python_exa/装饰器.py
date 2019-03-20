# -*- coding: utf-8 -*-

# @Time    : 2019-03-20 15:36
# @Author  : jian
# @File    : 装饰器.py

# 记录函数执行时间的装饰器

"""
初级装饰器
"""
import time


def record_time(func):
    def wrapper():
        start = time.clock()
        func()
        end = time.clock()
        print("----use---:", end - start)
        return func()

    return wrapper


@record_time
def foo():
    print("func run")


foo()


# 带参数的装饰器
def record_time(func):
    def wrapper(*args, **kwargs):
        start = time.clock()
        func(*args, **kwargs)
        end = time.clock()
        print("----use---:", end - start)
        return func(*args, **kwargs)

    return wrapper


@record_time
def foo(something):
    print("func run {}".format(something))


foo("say hello ")

"""
高级装饰器
"""


def logging(level):
    def wrapper(func):
        def inner_wrapper(*args, **kwargs):
            print("[{level}]: enter function {func}()".format(level = level, func = func.__name__))
            return func(*args, **kwargs)
        return inner_wrapper
    return wrapper


@logging(level='INFO')
def say(something):
    print("say {}!".format(something))


# 如果没有使用@语法，等同于
# say = logging(level='INFO')(say)

@logging(level='DEBUG')
def do(something):
    print(
    "do {}...".format(something))


say('hello')
do("my work")


"""
装饰器函数其实是这样一个接口约束，它必须接受一个callable对象作为参数，
然后返回一个callable对象。
在Python中一般callable对象都是函数，但也有例外。
只要某个对象重载了__call__()方法，那么这个对象就是callable的。
"""

class Test():
    def __call__(self):
        print( 'call me!')

t = Test()
t()
# call me

# 类装饰器
class logging(object):
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        print( "[DEBUG]: enter function {func}()".format(func=self.func.__name__))
        return self.func(*args, **kwargs)

@logging
def say_what(something):
    print( "say {}!".format(something))

say_what("hello my word")



# 带参数的类装饰器

class logging(object):
    def __init__(self, level ="INFO"):
        self.level = level

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            print( "[{level}]: enter function {func}()".format(level=self.level, func=func.__name__))
            func(*args, **kwargs)
        return wrapper


@logging(level='INFO')
def say_hehe(something):
    print("say {}!".format(something))

say_hehe("smile")


"""
内置装饰器
内置的装饰器和普通的装饰器原理是一样的，只不过返回的不是函数，而是类对象，所以更难理解一些
"""
#@property
# https://www.liaoxuefeng.com/wiki/001374738125095c955c1e6d8bb493182103fac9270762a000/001386820062641f3bcc60a4b164f8d91df476445697b9e000