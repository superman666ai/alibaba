# -*- coding: utf-8 -*-

# @Time    : 2019-04-15 10:43
# @Author  : jian
# @File    : 线性回归.py

"""
线性回归

线性：y=a*x 一次方的变化
回归：回归到平均值

简单线性回归
算法==公式
一元一次方程组

一元指的一个X：影响Y的因素，维度
一次指的X的变化：没有非线性的变化

y = a*x + b
x1,y1  x2,y2  x3,y3  x4,y4 ...

做机器学习，没有完美解
只有最优解~
做机器学习就是要以最快的速度，找到误差最小的最优解！

一个样本的误差：
yi^ - yi
找到误差最小的时刻，为了去找到误差最小的时刻，需要反复尝试，a,b
根据最小二乘法去求得误差
反过来误差最小时刻的a,b就是最终最优解模型！！！

多元线性回归
本质上就是算法（公式）变换为了多元一次方程组
y = w1*x1 + w2*x2 + w3*x3 + ... + wn*xn + w0*x0


Q：为什么求总似然的时候，要用正太分布的概率密度函数？
A：中心极限定理，如果假设样本之间是独立事件，误差变量随机产生，那么就服从正太分布！

Q：总似然不是概率相乘吗？为什么用了概率密度函数的f(xi)进行了相乘？
A：因为概率不好求，所以当我们可以找到概率密度相乘最大的时候，就相当于找到了概率相乘最大的时候！

Q：概率为什么不好求？
A：因为求得是面积，需要积分，麻烦，大家不用去管数学上如何根据概率密度函数去求概率！

Q：那总似然最大和最有解得关系？
A：当我们找到可以使得总似然最大的条件，也就是可以找到我们的DataSet数据集最吻合某个正太分布！
   即找到了最优解！

通过最大似然估计得思想，利用了正太分布的概率密度函数，推导出来了损失函数

Q：何为损失函数？
A：一个函数最小，就对应了模型是最优解！预测历史数据可以最准！

Q：线性回归的损失函数是什么？
A：最小二乘法，MSE，mean squared error，平方均值损失函数，均方误差

Q：线性回归的损失函数有哪些假设？
A：样本独立，随机变量，正太分布

通过对损失函数求导，来找到最小值，求出theta的最优解！

通过Python调用numpy来应用解析解公式之间计算最优解
文件名: linear_regression_0.py

讲解梯度下降法(重点内容)
1，初始化theta
2，求梯度gradients
3，调整theta
theta_t+1 = theta_t - grad*(learning_rate)
4，回到2循环往复第2步和第3步，直到迭代收敛，g约等于0

通过sklearn模块使用LinearRegression
from sklearn.linear_model import LinearRegression
文件名: linear_regression_1.py
"""


import numpy as np
import matplotlib.pyplot as plt

# 这里相当于是随机X维度X1，rand是随机均匀分布
X = 2 * np.random.rand(100, 1)

# # 人为的设置真实的Y一列，np.random.randn(100, 1)是设置error，randn是标准正太分布
y = 4 + 3 * X + np.random.randn(100, 1)
# # 整合X0和X1
X_b = np.c_[np.ones((100, 1)), X]
# print(X_b)
#
# # 常规等式求解theta
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
# print(theta_best)
#
# # 创建测试集里面的X1
X_new = np.array([[0], [2]])
X_new_b = np.c_[(np.ones((2, 1))), X_new]
print(X_new_b)
y_predict = X_new_b.dot(theta_best)
print(y_predict)

plt.plot(X_new, y_predict, 'r-')
plt.plot(X, y, 'b.')
plt.axis([0, 2, 0, 15])
plt.show()


#
#
import numpy as np
from sklearn.linear_model import LinearRegression


X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

lin_reg = LinearRegression()
lin_reg.fit(X, y)
print(lin_reg.intercept_, lin_reg.coef_)

X_new = np.array([[0], [2]])
print(lin_reg.predict(X_new))



