import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def decision():
    """
    决策树 对泰坦尼克进行预测生死
    :return:
    """

    # 读取数据
    data = pd.read_csv("data/titanic/train.csv")

    print(data.head(10))
    # 处理数据 找出特征值 目标值
    x = data[['Pclass', 'Age', 'Sex']]
    y = data['Survived']

    print(x.head(10))
    print(y.head(10))

    # 缺失值处理
    x['Age'].fillna(x['Age'].mean(), inplace=True)

    # 分割数据集到训练集测试机
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # 进行处理 （特征工程） 特征 》 类别 》 one_hot 编码
    dict = DictVectorizer(sparse=False)
    x_train = dict.fit_transform(x_train.to_dict(orient='records'))
    x_test = dict.transform(x_test.to_dict(orient='records'))

    rf = RandomForestClassifier()

    param = {"n_estimators": [120, 200, 300, 500, 800, 1200], "max_depth": [5, 8, 15, 25, 30]}
    # 网格搜索 交叉验证
    gc = GridSearchCV(rf, param_grid=param, cv=10)
    gc.fit(x_train, y_train)

    print("准确率：", gc.score(x_test, y_test))
    print("选择模型：", gc.best_params_)

    # # 用决策树进行预测
    # dec = DecisionTreeClassifier(max_depth=5)
    # dec.fit(x_train, y_train)
    #
    # # 预测准确率
    # print(dec.score(x_test, y_test))

    return None


if __name__ == "__main__":
    decision()
