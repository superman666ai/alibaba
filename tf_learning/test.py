# -*- coding: utf-8 -*-

# @Time    : 2019-04-04 8:48
# @Author  : jian
# @File    : test.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

points_num = 100

vectors = []
for i in range(points_num):
    x1 = np.random.normal(0.0, 0.66)
    y1 = 0.1 * x1 + 0.2 + np.random.normal(0.0, 0.04)
    vectors.append([x1, y1])

x_data = [v[0] for v in vectors]
y_data = [v[1] for v in vectors]

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b
# loss
loss = tf.reduce_mean(tf.square(y - y_data))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
for step in range(20):
    sess.run(train)
    print("step", step, "loss", sess.run(loss), "b", sess.run(b), "W", sess.run(W))


plt.plot(x_data, y_data, "r*", label="origin data")
plt.title("origin data")

plt.plot(x_data, sess.run(W)* x_data +sess.run(b), label="fitted line")
plt.legend()

plt.xlabel("x")
plt.ylabel("y")
plt.show()

sess.close()









# W = tf.Variable(2.0, dtype=tf.float32, name="weight")
# b = tf.Variable(1.0, dtype=tf.float32, name="bias")
# x = tf.placeholder(dtype=tf.float32, name="input")
#
# with tf.name_scope("output"):
#     y = W * x + b
#
# path = "log"
#
# # 初始化
# init = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     sess.run(init)
#     writer = tf.summary.FileWriter(path, sess.graph)
#     result = sess.run(y, {x: 3.0})
#     print(result)
