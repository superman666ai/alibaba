# -*- coding: utf-8 -*-

# @Time    : 2019-01-18 11:16
# @Author  : jian
# @File    : base_plt.py
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

"""
柱状图 histogram
线性图 linear
"""


def histogram():
    """

    :return:
    """
    k = 10
    x = np.arange(k)
    y = np.random.rand(k)

    # 画出 x 和 y 的柱状图
    plt.bar(x, y)

    # 增加数值
    for x, y in zip(x, y):
        plt.text(x, y, '%.2f' % y, ha='center', va='bottom')

    plt.show()


def linear():
    """

    :return:
    """
    # 设置图大小
    # plt.figure(figsize=(6, 3))
    x = np.linspace(0, 2 * np.pi, 50)
    y = np.sin(x)

    # plt.plot(x, y)
    # # 绘制第二条线
    # plt.plot(x, y * 2)

    # 调整样式 颜色  点线
    # lolor   b 绿色 g 红色 r 青色 c 品红 m 黄色 y 黑色 k 白色 w
    # linear 直线 - 虚线 - - 点线: 点划线 -.
    plt.plot(x, y, 'ys-', label="sin(x)")
    plt.plot(x, y * 2, 'm--')

    # 设置标题
    plt.title("sin(x) & 2sin(x)")

    # 设置坐标
    plt.xlim((0, np.pi + 1))
    plt.ylim((-3, 3))
    plt.xlabel('X')
    plt.ylabel('Y')

    # 设置刻度
    plt.xticks((0, np.pi * 0.5, np.pi, np.pi * 1.5, np.pi * 2))

    # 设置label
    plt.legend(loc='best')

    # 画出标注点
    x0 = np.pi
    y0 = 0
    plt.scatter(x0, y0, s=50)
    plt.annotate('sin(np.pi)=%s' % y0, xy=(np.pi, 0), xycoords='data', xytext=(+30, -30),
                 textcoords='offset points', fontsize=16,
                 arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2"))
    plt.text(0.5, -0.25, "sin(np.pi) = 0", fontdict={'size': 16, 'color': 'r'})

    """对于 annotate 函数的参数，做一个简单解释：
    'sin(np.pi)=%s' % y0 代表标注的内容，可以通过字符串 %s 将 y0 的值传入字符串；
    参数 xycoords='data' 是说基于数据的值来选位置;
    xytext=(+30, -30) 和 textcoords='offset points' 表示对于标注位置的描述 和 xy 偏差值，即标注位置是 xy 位置向右移动 30，向下移动30；
    arrowprops 是对图中箭头类型和箭头弧度的设置，需要用 dict 形式传入。

    """

    plt.show()


if __name__ == "__main__":
    histogram()
    # linear()
