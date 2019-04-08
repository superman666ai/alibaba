# -*- coding: utf-8 -*-

# @Time    : 2019-01-16 14:52
# @Author  : jian
# @File    : test_nn.py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tf_learning import input_data

mnist = input_data.read_data_sets("data/", one_hot=True)

trainimg = mnist.train.images
trainlabel = mnist.train.labels
testimg = mnist.test.images
testlabel = mnist.test.labels

print(trainimg.shape[0])
nsample = 5
randidx = np.random.randint(trainimg.shape[0], size=nsample)

for i in randidx:
    curr_img   = np.reshape(trainimg[i, :], (28, 28)) # 28 by 28 matrix
    curr_label = np.argmax(trainlabel[i, :] ) # Label
    plt.matshow(curr_img, cmap=plt.get_cmap('gray'))
    plt.title(str(i) + str(curr_label))
    plt.show()

# x = tf.placeholder("float", [None, 784])
# y = tf.placeholder("float", [None, 10])  # None is for infinite
# W = tf.Variable(tf.zeros([784, 10]))
# b = tf.Variable(tf.zeros([10]))
#
# actv = tf.nn.softmax(tf.matmul(x, W) + b)
# # COST FUNCTION
# cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(actv), reduction_indices=1))
# # OPTIMIZER
# learning_rate = 0.01
# optm = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
#
# pred = tf.equal(tf.argmax(actv, 1), tf.argmax(y, 1))
# # ACCURACY
# accr = tf.reduce_mean(tf.cast(pred, "float"))
# # INITIALIZER
# init = tf.global_variables_initializer()
#
# training_epochs = 50
# batch_size      = 100
# display_step    = 5
# # SESSION
# sess = tf.Session()
# sess.run(init)
# # MINI-BATCH LEARNING
# for epoch in range(training_epochs):
#     avg_cost = 0.
#     num_batch = int(mnist.train.num_examples/batch_size)
#     for i in range(num_batch):
#         batch_xs, batch_ys = mnist.train.next_batch(batch_size)
#         sess.run(optm, feed_dict={x: batch_xs, y: batch_ys})
#         feeds = {x: batch_xs, y: batch_ys}
#         avg_cost += sess.run(cost, feed_dict=feeds)/num_batch
#     # DISPLAY
#     if epoch % display_step == 0:
#         feeds_train = {x: batch_xs, y: batch_ys}
#         feeds_test = {x: mnist.test.images, y: mnist.test.labels}
#         train_acc = sess.run(accr, feed_dict=feeds_train)
#         test_acc = sess.run(accr, feed_dict=feeds_test)
#         print ("Epoch: %03d/%03d cost: %.9f train_acc: %.3f test_acc: %.3f"
#                % (epoch, training_epochs, avg_cost, train_acc, test_acc))
# print ("DONE")