#!/usr/bin/python3
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from scipy.stats import pearsonr
df = pd.read_csv("../../data/zhengqi_train.txt", sep='\t')

x_train = df[['V0','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20','V21','V22','V23','V24','V25','V26','V27','V28','V29','V30','V31','V32','V33','V34','V35','V36','V37']]
y_train = df[['target']]

dftest = pd.read_csv("../../data/zhengqi_test.txt", sep='\t')
x_test = dftest[['V0','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20','V21','V22','V23','V24','V25','V26','V27','V28','V29','V30','V31','V32','V33','V34','V35','V36','V37']]

def training(model, x_train, y_train, x_test):
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    carttree = DecisionTreeRegressor(max_depth=4)
    carttree.fit(x_train, y_train)
    y_test = carttree.predict(x_test)
    return y_test

def pearsoncal(x_train, y_train):
    #return pearsonr(x_train, y_train)
    #return np.corrcoef(x_train,rowvar=0)
    return np.corrcoef(x_train,rowvar=0)

if __name__ == "__main__":

    #print(pearsoncal(df, y_train).shape)
    #print(pearsoncal(df, y_train))
    colmat = pearsoncal(df, y_train)
    #print(type(colmat[-1]))
    colval = colmat[-1].tolist()
    #print((colmat[-1]).tolist())
    feature = [True if item > 0.5 else False for item in colval ]
    feature = feature[:-1]

    x_train_fea = x_train.iloc[:, feature]
    x_test_fea = x_test.iloc[:, feature]
    y_test = training("cart", x_train_fea, y_train, x_test_fea)
    np.savetxt('../result/carttree.txt', y_test)
