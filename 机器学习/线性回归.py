from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def myLinear():
    """
    线性回归预测波士顿房子价格
    :return:
    """

    # 1 获取数据
    lb = load_boston()

    # 2 分割数据集
    x_train, x_test, y_train, y_test = train_test_split(lb.data, lb.target, test_size=0.25)

    print(y_test)

    # 3 标准化处理
    # 特征值 目标值都要进行标准化  实例化两个标准化API
    std_x = StandardScaler()
    x_train = std_x.fit_transform(x_train)
    x_test = std_x.transform(x_test)

    std_y = StandardScaler()
    y_train = std_y.fit_transform(y_train.reshape(-1, 1))
    # y_test = std_y.transform(y_test.reshape(1, -1))

    # 4 预测 正规方程预测
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    print(lr.coef_)

    # 预测测试集的房子价格
    y_predict = lr.predict(x_test)
    print(std_y.inverse_transform(y_predict).reshape(1, -1))

    # 4 预测 梯度下降预测
    sgd = SGDRegressor()
    sgd.fit(x_train, y_train)
    print(sgd.coef_)

    # 预测测试集的房子价格
    y_predict = sgd.predict(x_test)
    print(std_y.inverse_transform(y_predict).reshape(1, -1))

    return None


if __name__ == "__main__":
    myLinear()
