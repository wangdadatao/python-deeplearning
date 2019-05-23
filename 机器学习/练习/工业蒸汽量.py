from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np


def industrial():
    """
    线性回归 预测工业蒸汽量
    :return:
    """
    # 1 读取数据
    # 训练数据
    data_train = pd.read_csv("../data/zhengqi/zhengqi_train.csv")
    train = data_train.drop(['target'], axis=1)
    tr_target = data_train['target']

    # 预测数据
    data_test = pd.read_csv("../data/zhengqi/zhengqi_test.csv")
    test = data_test

    # 3 标准化处理
    # 特征值 目标值都要进行标准化  实例化两个标准化API
    std_x = StandardScaler()
    x_train = std_x.fit_transform(train)
    x_test = std_x.transform(test)

    std_y = StandardScaler()
    y_train = std_y.fit_transform(tr_target.values.reshape(-1, 1))

    # 4 预测 正规方程预测
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    # print(lr.coef_)

    # 预测测试集的房子价格
    y_predict = std_y.inverse_transform(lr.predict(x_test))

    print(y_predict.reshape(-1,).size,data_test.__len__())

    data_test.insert(0, 'target', y_predict.reshape(-1,))
    data_test.to_csv('../data/zhengqi/zhengqi_test.csv', index=False, encoding='utf-8')
    print(y_predict)

    # # print(std_y.inverse_transform(y_test).reshape(-1, 1), y_predict)
    # # print(y_test.reshape(-1, 1))
    #
    # y_test = std_y.inverse_transform(y_test)
    # print("均方误差", mean_squared_error(y_test.reshape(-1, 1), y_predict))
    #
    # re = np.array([y_test.reshape(-1, ), y_predict.reshape(-1, )])
    # print(re)

    return None


if __name__ == "__main__":
    industrial()
