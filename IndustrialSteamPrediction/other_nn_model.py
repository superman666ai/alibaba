# -*- encoding:utf-8 -*-
import tensorflow as tf
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split

boston = load_boston()
# X = scale(boston.data)
# y = scale(boston.target.reshape((-1,1)))


X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.1, random_state=0)



X_train = scale(X_train)
X_test = scale(X_test)
y_train = scale(y_train.reshape((-1, 1)))
y_test = scale(y_test.reshape((-1, 1)))

print(X_train, type(X_train), X_test.shape)