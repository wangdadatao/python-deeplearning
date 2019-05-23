from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_20newsgroups, load_iris
from sklearn.feature_extraction.text import TfidfVectorizer


def decision():
    """
    决策树 对泰坦尼克进行预测生死
    :return:
    """

    # 1 读取数据
    data = load_iris()

    print(data)

    # 进行数据的分割 训练集 测试机
    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.25)

    # 3 特征工程（标准化）
    std = StandardScaler()
    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)

    # 用决策树进行预测
    dec = DecisionTreeClassifier()
    dec.fit(x_train, y_train)

    # 预测准确率
    print(dec.score(x_test, y_test))

    return None


if __name__ == "__main__":
    decision()
