from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


def decision_tree():
    """
    决策树
    :return:
    """

    # 1 读取数据
    titans_train = pd.read_csv("../data/titanic/train.csv")
    titans_test = pd.read_csv("../data/titanic/test.csv")
    sub = pd.read_csv("../data/titanic/gender_submission.csv")

    # 处理数据 找出特征值 目标值
    x_train = titans_train[['Pclass', 'Age', 'Sex']]
    x_target = titans_train['Survived']

    y_test = titans_test[['Pclass', 'Age', 'Sex']]

    # 缺失值处理
    x_train['Age'].fillna(x_train['Age'].mean(), inplace=True)
    y_test['Age'].fillna(y_test['Age'].mean(), inplace=True)

    # print(x.head(10))
    print(y_test.head(10))

    # 2 进行数据的分割 训练集 测试机
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # 进行处理 （特征工程） 特征 》 类别 》 one_hot 编码
    dict = DictVectorizer(sparse=False)
    x_train = dict.fit_transform(x_train.to_dict(orient='records'))
    y_test = dict.transform(y_test.to_dict(orient='records'))

    # 3 用决策树进行预测
    dec = DecisionTreeClassifier()
    dec.fit(x_train, x_target)

    rf = RandomForestClassifier(n_estimators=120, max_depth=5)
    rf.fit(x_train)

    y_predict = rf.predict(y_test)
    print(y_predict)

    # 网格搜索 交叉验证
    param = {"n_estimators": [120], "max_depth": [5]}
    gc = GridSearchCV(rf, param_grid=param, cv=2)
    gc.fit(x_train, x_target)

    y_predict = gc.predict(y_test)

    sub.insert(0, 'Survived4', y_predict)
    print(sub.head(10))
    sub.to_csv('../data/titanic/gender_submission.csv', index=False, encoding='utf-8')

    return None


if __name__ == "__main__":
    decision_tree()
