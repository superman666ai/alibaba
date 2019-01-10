# -*- encoding:utf-8 -*-
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv('data/zhengqi_train.txt', sep='\t')

# 测试模型的数据集
test_df = pd.read_csv('data/zhengqi_test.txt', sep='\t')
test_df.drop(['V5', 'V17', 'V28', 'V22', 'V11', 'V9'], axis=1, inplace=True)

# 剔除认为不重要的特征
df.drop(['V5', 'V17', 'V28', 'V22', 'V11', 'V9'], axis=1, inplace=True)

x = df.iloc[:, :-1]

y = df.target

# 标准化特征

mm = MinMaxScaler()

x = mm.fit_transform(x)

# 数据集划分

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)


# 简单线性回归
def linear_regression(x_train, x_test, y_train, y_test):
    """

    :param x_train:
    :param x_test:
    :param y_train:
    :param y_test:
    :return:
    """
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    score = lr.score(x_test, y_test)

    print(score)

    y_pred = lr.predict(x_test)

    print(mean_squared_error(y_test, y_pred))

    result = lr.predict(test_df)
    print(result)

    # 保存结果
    result_df = pd.DataFrame(result, columns=['target'])
    result_df.to_csv("0.098.txt", index=False, header=False)


linear_regression(x_train, x_test, y_train, y_test)
