# -*- coding: utf-8 -*-

# @Time    : 2019-03-27 16:20
# @Author  : jian
# @File    : 字典.py
dic = {"name": "axiaoli"}
dic2 = {"age": "fdsa"}
# 添加键 返回none
dic.update(dic2)

# 删除键 不存在会报key error
# del dic["name"]
# x[1] 根据值排序  x[0] 根据键排序
dic = sorted(dic.items(), key=lambda x:x[1], reverse=False)
# print(dict(dic))

