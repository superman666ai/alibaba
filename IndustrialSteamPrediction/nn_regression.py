# -*- encoding:utf-8 -*-
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

import numpy as np

df = pd.read_csv('data/zhengqi_train.txt', sep='\t')

test_df = pd.read_csv('data/zhengqi_test.txt', sep='\t')

# 数据处理

x = df.iloc[:, :-1]
y = df["target"]
y = np.array(y)[:, np.newaxis]

# 数据集划分
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

x_train = scale(x_train)
x_test = scale(x_test)

y_train = scale(y_train)
y_test = scale(y_test)


# 添加层

# 创建一个神经网络层

def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


xs = tf.placeholder(shape=[None, x_train.shape[1]], dtype=tf.float32, name="inputs")

ys = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="y_true")

keep_prob_s = tf.placeholder(dtype=tf.float32)

l1 = add_layer(xs, 38, 10, activation_function=tf.nn.relu)

pred = add_layer(l1, 10, 1)

pred = tf.add(pred, 0, name='pred')

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - pred), reduction_indices=[1]))  # mse

tf.summary.scalar("loss", tensor=loss)

train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()


keep_prob=1  # 防止过拟合，取值一般在0.5到0.8。我这里是1，没有做过拟合处理
ITER =5000  # 训练次数


feed_dict_train = {ys: y, xs: X, keep_prob_s: keep_prob}


with tf.Session() as sess:
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(logdir="nn_boston_log", graph=sess.graph)  # 写tensorbord
    sess.run(init)
    for i in range(n):
        _loss, _ = sess.run([loss, train_op], feed_dict=feed_dict_train)

        if i % 100 == 0:
            print("epoch:%d\tloss:%.5f" % (i, _loss))
            y_pred = sess.run(pred, feed_dict=feed_dict_train)
            rs = sess.run(merged, feed_dict=feed_dict_train)
            writer.add_summary(summary=rs, global_step=i)  # 写tensorbord

# l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
#
# prediction = add_layer(l1, 10, 1, activation_function=None)
