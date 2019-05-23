from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np


def logistic():
    """
    逻辑回归 也测癌症率
    :return:
    """

    data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
    names = ['Sample code number', 'Clump Thickness',
             'Uniformity of Cell Size', 'Uniformity of Cell Shape',
             'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei',
             'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'target']

    data = pd.read_csv(data_url, names=names)

    data = data.replace(to_replace="?", value=np.nan).dropna()

    train = data.drop(["target", "Sample code number"], axis=1)
    target = data["target"]

    # 数据分割
    x_train, x_test, y_train, y_test = train_test_split(train, target, test_size=0.25)

    # 标准化
    std = StandardScaler()
    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)

    # 进行逻辑回归
    lg = LogisticRegression(C=1.0)
    lg.fit(x_train, y_train)

    print(lg.coef_)
    print("准确率：", lg.score(x_test, y_test))

    return None


if __name__ == "__main__":
    logistic()
