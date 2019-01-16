# -*- encoding:utf-8 -*-
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from utils import save_result
import numpy as np

df = pd.read_csv('data/zhengqi_train.txt', sep='\t')

test_df = pd.read_csv('data/zhengqi_test.txt', sep='\t')

# 剔除认为不重要的特征
# df.drop(['V5', 'V17', 'V28', 'V22', 'V11', 'V9'], axis=1, inplace=True)
# test_df.drop(['V5', 'V17', 'V28', 'V22', 'V11', 'V9'], axis=1, inplace=True)

# 数据处理

x = df.iloc[:, :-1]
y = df["target"]

y = np.array(y)[:, np.newaxis]

x_mm = MinMaxScaler()
x = x_mm.fit_transform(x)
test_df = x_mm.fit_transform(test_df)

y_mm = MinMaxScaler()
y = y_mm.fit_transform(y)

# 数据集划分
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)


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
drop_out = 0.6

l1 = add_layer(xs, x_train.shape[1], 100, activation_function=tf.nn.sigmoid)
# l1 = add_layer(xs, x_train.shape[1], 100, activation_function=tf.nn.tanh)
# l1 = add_layer(xs, x_train.shape[1], 100, activation_function=tf.nn.relu)
l1 = layer1 = tf.nn.dropout(l1, drop_out)

# l2 = add_layer(l1, 100, 10, activation_function=tf.nn.sigmoid)
l2 = add_layer(l1, 100, 10, activation_function=tf.nn.tanh)
# l2 = add_layer(l1, 100, 10, activation_function=tf.nn.relu)
l2 = tf.nn.dropout(l2, drop_out)

prediction = add_layer(l2, 10, 1, activation_function=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                    reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

init = tf.global_variables_initializer()

feed_dict_train = {ys: y_train, xs: x_train}

# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)

    # Fit all training data
    for i in range(1000000):

        sess.run(train_step, feed_dict=feed_dict_train)
        if i % 50 == 0:
            a = sess.run(loss, feed_dict={xs: x_test, ys: y_test})
            print(a, type(a))

            # 当误差达到阈值 储存结果

            if float(a) < 0.02:
                y_pre = sess.run(prediction, feed_dict={xs: test_df})
                y_pre = y_mm.inverse_transform(y_pre)
                pre = pd.DataFrame(y_pre, columns=["0"])
                pre = pre["0"]
                save_result(list(pre))

"""
画出平面图形 
# plot the real data
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_train, y_train)
plt.ion()#本次运行请注释，全局运行不要注释
plt.show()


"""
