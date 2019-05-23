from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB


def decision_tree():
    """
    决策树
    :return:
    """

    # 1 读取数据
    data = load_iris()

    # 2 进行数据的分割 训练集 测试机
    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.25)

    # 3 用决策树进行预测
    dec = DecisionTreeClassifier()
    dec.fit(x_train, y_train)

    # 预测准确率
    print(dec.score(x_test, y_test))

    return None


def knncls():
    """
    K 近邻
    :return:
    """
    # 1 读取数据
    data = load_iris()

    # 2 进行数据的分割 训练集 测试机
    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.25)

    # 3 特征工程（标准化）
    std = StandardScaler()
    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)

    # 4 进行算法流程
    knn = KNeighborsClassifier(n_neighbors=8)
    knn.fit(x_train, y_train)

    # 得出准确率
    print(knn.score(x_test, y_test))

    return None


def bayes():
    """
    朴素贝叶斯
    :return:
    """

    # 1 读取数据
    data = load_iris()

    # 2 进行数据的分割 训练集 测试机
    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.25)

    # 3 进行朴素贝叶斯算法
    mlt = MultinomialNB(alpha=1.0)
    mlt.fit(x_train, y_train)

    print(mlt.score(x_test, y_test))
    return None


if __name__ == "__main__":
    decision_tree()
    knncls()
    bayes()
