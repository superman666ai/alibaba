# -*- coding: utf-8 -*-

# @Time    : 2019-03-27 16:20
# @Author  : jian
# @File    : 字典.py
dic = {"name": "xiaoli"}
dic2 = {"age": 18}
# 添加键 返回none
dic.update(dic2)

# 删除键 不存在会报key error
del dic["name"]
print(dic)
