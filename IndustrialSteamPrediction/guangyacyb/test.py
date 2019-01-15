#!/usr/bin/python3
import pandas as pd
import numpy as np
import sklearn.linear_model as linear_model
from scipy.stats import pearsonr

df = pd.read_csv("../data/zhengqi_train.txt", sep='\t')


x_train = df.iloc[:, :-1]

y_train = df.target

x_test = pd.read_csv("../data/zhengqi_test.txt", sep='\t')



def training(model, x_train, y_train, x_test):
    # print(x_train.shape)
    # print(y_train.shape)
    # print(x_test.shape)
    if model == "linearregresion":
        linreg = linear_model.LinearRegression()
        linreg.fit(x_train, y_train)
        y_test = linreg.predict(x_test)
        return y_test
    if model == "ringergresion":
        linreg = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0])
        linreg.fit(x_train, y_train)
        y_test = linreg.predict(x_test)
        # print(linreg.coef_)
        return y_test

def pearsoncal(x_train, y_train):
    return np.corrcoef(x_train, rowvar=0)


if __name__ == "__main__":
    # print(pearsoncal(df, y_train).shape)
    # print(pearsoncal(df, y_train))
    colmat = pearsoncal(df, y_train)

    # print(type(colmat[-1]))
    colval = colmat[-1].tolist()
    # print((colmat[-1]).tolist())
    feature = [True if item > 0.5 else False for item in colval]
    feature = feature[:-1]

    # y_test = training("linearregresion", x_train, y_train, x_test)
    x_train_fea = x_train.iloc[:, feature]
    x_test_fea = x_train.iloc[:, feature]
    # y_test = training("ringergresion", x_train, y_train, x_test)

    #print(type(colmat[-1]))
    colval = colmat[-1].tolist() # 最后一行，是每个特征与目标值的相关系数
    #print((colmat[-1]).tolist())
    feature = [True if item > 0.5 else False for item in colval ] #只保留相关系数大于0.5的
    feature = feature[:-1]
    x_train_fea = x_train.iloc[:, feature]
    x_test_fea = x_train.iloc[:, feature]
    #y_test = training("linearregresion", x_train, y_train, x_test)
    #y_test = training("ringergresion", x_train, y_train, x_test)
    
    y_test = training("ringergresion", x_train_fea, y_train, x_test_fea)
    np.savetxt('./result.txt', y_test)
