#!/usr/bin/python3
import pandas as pd
import numpy as np
import sklearn.linear_model as linear_model
from scipy.stats import pearsonr
df = pd.read_csv("../data/zhengqi_train.txt", sep='\t')

# 训练集
x_train = df[['V0','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20','V21','V22','V23','V24','V25','V26','V27','V28','V29','V30','V31','V32','V33','V34','V35','V36','V37']]
y_train = df[['target']]
 
# 测试集
dftest = pd.read_csv("../data/zhengqi_test.txt", sep='\t')
x_test = dftest[['V0','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20','V21','V22','V23','V24','V25','V26','V27','V28','V29','V30','V31','V32','V33','V34','V35','V36','V37']]
 
def training(model, x_train, y_train, x_test):
    #print(x_train.shape)
    #print(y_train.shape)
    #print(x_test.shape)
    if model == "linearregresion":
        linreg = linear_model.LinearRegression()
        linreg.fit(x_train, y_train)
        y_test = linreg.predict(x_test)
        return y_test
    if model == "ringergresion":
        linreg = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0])
        linreg.fit(x_train, y_train)
        y_test = linreg.predict(x_test)
        #print(linreg.coef_)
        return y_test
 
#皮尔逊相关
def pearsoncal(x_train, y_train):
    return np.corrcoef(x_train,rowvar=0)
 
if __name__ == "__main__":
 
    #print(pearsoncal(df, y_train).shape)
    #print(pearsoncal(df, y_train))
    colmat = pearsoncal(df, y_train)
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
