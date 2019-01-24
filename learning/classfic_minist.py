# -*- coding: utf-8 -*-

# @Time    : 2019-01-24 15:37
# @Author  : jian
# @File    : classfic_minist.py

import tensorflow as tf
import numpy as np
import os
import input_data


# 准备数据库（MNIST库，这是一个手写体数据库）
mnist = input_data.read_data_sets('data/', one_hot=True)


def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    # 注意tensorflow进行运算时总是把数据都化为二维矩阵来运算
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    # activation_function is None时没有激励函数，是线性关系
    else:
        outputs = activation_function(Wx_plus_b)
    # activation_function不为None时，得到的Wx_plus_b再传入activation_function再处理一下
    return outputs


def compute_accuracy(v_xs, v_ys):
    global prediction
    # 使用global则对全局变量prediction进行操作
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    # 使用xs输入数据生成预测值prediction
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    # 对于预测值和真实值的差别
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # 计算预测的准确率
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    # run得到结果，结果是个百分比
    return result


xs = tf.placeholder(tf.float32, [None, 784])
# None表示不规定样本的数量，784表示每个样本的大小为28X28=784个像素点
ys = tf.placeholder(tf.float32, [None, 10])
# 每张图片表示一个数字，我们的输出是数字0到9，所以是10个输出

prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)
# 调用add_layer定义输出层，输入数据是784个特征，输出数据是10个特征，激励采用softmax函数
# softmax激励函数一般用于classification

# 搭建分类模型时，loss函数（即最优化目标函数）选用交叉熵函数（cross_entropy）
# 交叉熵用来衡量预测值和真实值的相似程度，如果完全相同，它们的交叉熵等于零。
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
# 定义训练函数,使用梯度下降法训练,0.5是学习效率，通常小于1,minimize(cross_entropy)指要将cross_entropy减小
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# 创建会话，并开始将网络初始化
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    # 训练时只从数据集中取100张图片来训练
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    if i % 50 == 0:
        # 每训练50次打印准确度
        print(compute_accuracy(mnist.test.images, mnist.test.labels))
# 对比mnist中的training data和testing data的准确度

