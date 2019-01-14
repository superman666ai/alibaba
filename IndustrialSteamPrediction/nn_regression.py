# -*- encoding:utf-8 -*-
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np

df = pd.read_csv('data/zhengqi_train.txt', sep='\t')

test_df = pd.read_csv('data/zhengqi_test.txt', sep='\t')

# 数据处理

# 剔除认为不重要的特征
df.drop(['V5', 'V17', 'V28', 'V22', 'V11', 'V9'], axis=1, inplace=True)
test_df.drop(['V5', 'V17', 'V28', 'V22', 'V11', 'V9'], axis=1, inplace=True)

x = df.iloc[:, :-1]
y = df.target

# 标准化特征
mm = MinMaxScaler()
x = mm.fit_transform(x)

# 结果集标准
test_df = mm.transform(test_df)

# 数据集划分
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)


# 添加层

# 创建一个神经网络层
def add_layer(input, in_size, out_size, activation_function=None):
    """
    :param input: 数据输入
    :param in_size: 输入大小（前一层神经元个数）
    :param out_size: 输出大小（本层神经元个数）
    :param activation_function: 激活函数（默认没有）
    :return:
    """
    Weight = tf.Variable(tf.random_normal([in_size, out_size]))

    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    W_mul_x_plus_b = tf.matmul(input, Weight) + biases

    if activation_function == None:

        output = W_mul_x_plus_b
    else:
        output = activation_function(W_mul_x_plus_b)


# 1.训练的数据

# print(x_train.shape)

# y_data = y_train

# 2.定义节点准备接收数据

xs = tf.placeholder(tf.float32, [None, 32])
ys = tf.placeholder(tf.float32, [None, 1])

# 3.定义神经层：隐藏层和预测层
# add hidden layer 输入值是 xs，在隐藏层有 10 个神经元

l1 = add_layer(xs, 1, 32, activation_function=tf.nn.relu)

# add output layer 输入值是隐藏层 l1，在预测层输出 1 个结果
# prediction = add_layer(l1, 10, 1, activation_function=None)

# 4.定义 loss 表达式
# the error between prediciton and real data
# loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
# reduction_indices=[1]))

# 5.选择 optimizer 使 loss 达到最小
# 这一行定义了用什么方式去减少 loss，学习率是 0.1
# train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)


# important step 对所有变量进行初始化
# init = tf.initialize_all_variables()
# sess = tf.Session()
# 上面定义的都没有运算，直到 sess.run 才会开始运算
# sess.run(init)

# 迭代 1000 次学习，sess.run optimizer
# for i in range(1000):
# training train_step 和 loss 都是由 placeholder 定义的运算，所以这里要用 feed 传入参数
# sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
# if i % 50 == 0:
# to see the step improvement
# print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
