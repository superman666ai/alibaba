# -*- coding: utf-8 -*-

# @Time    : 2019-03-28 14:20
# @Author  : jian
# @File    : 排序算法.py

# 根据排序过程中借助的主要操作，可把内排序分为：插入排序 交换排序 选择排序 归并排序

"""
冒泡排序
时间复杂度O(n^2)
交换排序的一种。
其核心思想是：两两比较相邻记录的关键字，如果反序则交换，直到没有反序记录为止。
其实现细节可以不同，比如下面3种：

最简单排序实现:bubble_sort_simple
冒泡排序:bubble_sort
改进的冒泡排序:bubble_sort_advance
"""


class SQList:
    def __init__(self, lis=None):
        self.r = lis

    def swap(self, i, j):
        """定义一个交换元素的方法，方便后面调用。"""
        self.r[i], self.r[j] = self.r[j], self.r[i]

    def bubble_sort_simple(self):
        """
        最简单的交换排序，时间复杂度O(n^2)
        """
        lis = self.r
        length = len(self.r)
        for i in range(length - 1):
            for j in range(i + 1, length):
                if lis[i] > lis[j]:
                    self.swap(i, j)

    def bubble_sort(self):
        """
        冒泡排序，时间复杂度O(n^2)
        """
        lis = self.r
        length = len(self.r)
        for i in range(length):
            j = length - 2
            while j >= i:
                if lis[j] > lis[j + 1]:
                    self.swap(j, j + 1)
                j -= 1

    def bubble_sort_advance(self):
        """
        冒泡排序改进算法，时间复杂度O(n^2)
        设置flag，当一轮比较中未发生交换动作，则说明后面的元素其实已经有序排列了。
        对于比较规整的元素集合，可提高一定的排序效率。
        """
        lis = self.r
        length = len(self.r)
        flag = True
        i = 0
        while i < length and flag:
            flag = False
            j = length - 2
            while j >= i:
                if lis[j] > lis[j + 1]:
                    self.swap(j, j + 1)
                    flag = True
                j -= 1
            i += 1

    def __str__(self):
        ret = ""
        for i in self.r:
            ret += " %s" % i
        return ret

if __name__ == '__main__':
    # sqlist = SQList([7, 1, 3, 4, 8, 5, 9, 2, 6])
    sqlist = SQList([2, 1, 4, 5, 7])
    # sqlist.bubble_sort_simple()
    # sqlist.bubble_sort()
    sqlist.bubble_sort_advance()
    print(sqlist)
