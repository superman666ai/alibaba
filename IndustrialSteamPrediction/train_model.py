# -*- encoding:utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('data/zhengqi_train.txt', sep='\t')

# 结果集
test_df = pd.read_csv('data/zhengqi_test.txt', sep='\t')

# 数据处理

# 剔除认为不重要的特征
# df.drop(['V5', 'V17', 'V28', 'V22', 'V11', 'V9'], axis=1, inplace=True)
# test_df.drop(['V5', 'V17', 'V28', 'V22', 'V11', 'V9'], axis=1, inplace=True)

x = df.iloc[:, :-1]
y = df.target

# 标准化特征
mm = MinMaxScaler()
x = mm.fit_transform(x)

# 结果集标准
test_df = mm.transform(test_df)

# 数据集划分
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)


# 简单回归模型
def simple_regression(x_train, x_test, y_train, y_test):
    # 线性回归
    # model = LinearRegression()

    # 岭回归
    model = Ridge(alpha=0.1)

    # lasso 回归
    # model = Lasso(alpha=0.1)

    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)

    print("score", score)
    y_pred = model.predict(x_test)

    print("mean_squared_error", mean_squared_error(y_test, y_pred))

    # result = model.predict(test_df)
    # print(result)

    # 保存结果
    # result_df = pd.DataFrame(result, columns=['target'])
    # result_df.to_csv("0.098.txt", index=False, header=False)


# 参数调优
def model_gridsearch(x_train, x_test, y_train, y_test):
    # 岭回归
    # model = Ridge()

    # lasso 回归
    model = Lasso()

    param = {}
    param["alpha"] = np.arange(0.1, 0.6, 0.1)
    gc = GridSearchCV(model, param_grid=param, cv=5)
    gc.fit(x_train, y_train)

    # 预测准确率
    print(gc.score(x_test, y_test))

    # 交叉验证中最好的结果
    print(gc.best_score_)

    # 最好的模型
    print(gc.best_estimator_)

    # 每个k的 验证结果
    print(gc.cv_results_)



    # result = model.predict(test_df)
    # print(result)

    # 保存结果
    # result_df = pd.DataFrame(result, columns=['target'])
    # result_df.to_csv("0.098.txt", index=False, header=False)



if __name__ == "__main__":
    # 简单回归
    simple_regression(x_train, x_test, y_train, y_test)

    # model_gridsearch(x_train, x_test, y_train, y_test)
    # Ridge Lasso 回归alpha=0.1 比较好

