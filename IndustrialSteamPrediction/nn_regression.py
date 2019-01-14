# -*- encoding:utf-8 -*-
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np

df = pd.read_csv('data/zhengqi_train.txt', sep='\t')

test_df = pd.read_csv('data/zhengqi_test.txt', sep='\t')

# 数据处理

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


from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

# sess = tf.InteractiveSession()
#
# x = tf.placeholder(tf.float32,[None,784])
# w = tf.Variable(tf.zeros([784,10]))
# b = tf.Variable(tf.zeros([10]))
#
# y_pre = tf.nn.softmax(tf.matmul(x,w) + b) #预测值,预测标签
#
# y_true = tf.placeholder(tf.float32,[None,10])#标签
# cross_entropy = tf.reduce_mean(tf.reduce_sum(-y_true * tf.log(y_pre), reduction_indices=[1]))
#
# train = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#
# tf.global_variables_initializer().run()
#
#
# for i in range(1000):
#     batch_x,batch_y = mnist.train.next_batch(100)
#     sess.run(train,feed_dict={x:batch_x,y_true:batch_y})
# correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(y_true,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
# print(accuracy.eval({x: mnist.test.images,y_true: mnist.test.labels}))

