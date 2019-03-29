# -*- coding: utf-8 -*-

# @Time    : 2019-03-28 10:59
# @Author  : jian
# @File    : +=操作.py
"""
+= 操作调用__iadd__方法，没有该方法时，再尝试调用__add__方法
__iadd__方法直接在原对象a1上进行更新，该方法的返回值为None
__add__方法会返回一个新的对象，原对象不修改，
因为这里 a1被重新赋值了，a1指向了一个新的对象，所以出现了文章开头a1不等于a2的情况

"""
a1 = list(range(3))
a2 = a1
a2 += [3]
# print(a1)
# print(id((a1)))
# [0, 1, 2, 3]
# print(a2)
# print(id((a2)))
# [0, 1, 2, 3]


a1 = list(range(3))
a2 = a1
a2 = a2 + [3]


# print(a1)
# print(id((a1)))
# [0, 1, 2]
# print(a2)
# print(id((a2)))
# [0, 1, 2, 3]




