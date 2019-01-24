# -*- coding: utf-8 -*-

# @Time    : 2018/11/23 13:59
# @Author  : jian
# @File    : class_model_nn.py
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv("data/data.csv")
# 构建特征 目标值

x = data.iloc[0: -1, 1:7]
y = data.iloc[1:, 7:]
labels = []
for i in range(1, 17):
    list = []
    list.append(i)
    labels.append(list)

onehot = OneHotEncoder(sparse=False)
onehot.fit_transform(labels)
y = onehot.transform(y)


# 标准化数据 minmax
x_mm = MinMaxScaler()
x = x_mm.fit_transform(x)

# 数据集划分
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

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
ys = tf.placeholder(shape=[None, 16], dtype=tf.float32, name="y_true")
drop_out = 0.6

l1 = add_layer(xs, x_train.shape[1], 50, activation_function=tf.nn.sigmoid)
l1 = layer1 = tf.nn.dropout(l1, drop_out)
#
#
l2 = add_layer(l1, 50, 20, activation_function=tf.nn.sigmoid)
l2 = tf.nn.dropout(l2, drop_out)

prediction = add_layer(l2, 20, 16, activation_function=None)


# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits))
loss = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))

train_step = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)

init = tf.global_variables_initializer()

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

# Start training


sess = tf.Session()
sess.run(init)


for i in range(100000):
    batch_xs, batch_ys = x_train, y_train
    # 训练时只从数据集中取100张图片来训练
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    if i % 50 == 0:
        # 每训练50次打印准确度
        print(compute_accuracy(x_test, y_test))
#
    #
    # # Fit all training data
    # for i in range(1000000):
    #     # sess.run(train_step, feed_dict=feed_dict_train)
    #
    #     # print(sess.run(prediction, feed_dict=feed_dict_train))
    #     if i % 50 == 0:
    #         train_acc = sess.run(loss, feed_dict=feed_dict_train)
    #         print("TRAIN ACCURACY:",train_acc)
    #
    #         feeds = {xs: x_test, ys: y_test}
    #         test_acc = sess.run(loss, feed_dict=feeds)
    #         print("TEST ACCURACY:", test_acc)
            #
            # if float(test_acc) < 0.2:
            #     y_pre = sess.run(prediction, feed_dict={xs: test_df})
            #     y_pre = y_mm.inverse_transform(y_pre)
            #     pre = pd.DataFrame(y_pre, columns=["0"])
            #     pre = pre["0"]
            #     save_result(list(pre))

#
#
# dict = nn.predict(x_test)
# print(nn.out_activation_)
#
# 预测准确率
# print(nn.score(x_test, y_test))
#
# t = np.arange(len(y_test))
# plt.figure()
# plt.plot(t, y_test, 'r-', linewidth=2, label='Test_b')
# plt.plot(t, y_predict, 'g-', linewidth=2, label='Predict_b')
# plt.xticks(tuple(x for x in range(len(y_test))))
# plt.grid()
# plt.show()
