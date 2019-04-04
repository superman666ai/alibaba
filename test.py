# -*- coding: utf-8 -*-

# @Time    : 2019-04-04 8:48
# @Author  : jian
# @File    : test.py

import tensorflow as tf
import tensorboard as tb

W = tf.Variable(2.0, dtype=tf.float32, name="weight")
b = tf.Variable(1.0, dtype=tf.float32, name="bias")
x = tf.placeholder(dtype=tf.float32, name="input")

with tf.name_scope("output"):
    y = W * x + b

path = "log"

# 初始化
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter(path, sess.graph)
    result = sess.run(y, {x: 3.0})
    print(result)


