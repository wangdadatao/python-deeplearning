from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Imputer
from sklearn.feature_selection import VarianceThreshold
import numpy as np
import jieba


def dectvec():
    dict = DictVectorizer(sparse=True)

    data = dict.fit_transform([
        {
            "city": "hangzhou",
            "num": 89
        },
        {
            "city": "shanghai",
            "num": 39
        },
        {
            "city": "beijing",
            "num": 78
        },
        {
            "city": "hangzhou",
            "num": 84
        }
    ])

    print(data)
    print(dict.get_feature_names())
    print(dict.inverse_transform(data))
    return None


def countvec():
    cv = CountVectorizer()

    date = cv.fit_transform([
        "how are you are ?",
        "how are you too !"
    ])

    print(cv.get_feature_names())
    print(date.toarray())

    return None


def cut():
    str1 = jieba.cut("你吃过饭了吗？这个蛋糕很好吃。")
    str2 = jieba.cut("我不喜欢吃饭，我昨天吃了三明治。")
    str3 = jieba.cut("这个风吹的好舒服，我想吃冰淇淋。")

    # 转成列表
    strList1 = list(str1)
    strList2 = list(str2)
    strList3 = list(str3)

    # 转成字符串
    r1 = " ".join(strList1)
    r2 = " ".join(strList2)
    r3 = " ".join(strList3)

    return r1, r2, r3


def zhongwenvec():
    cv = CountVectorizer()
    r1, r2, r3 = cut()
    date = cv.fit_transform([
        r1, r2, r3
    ])

    print(cv.get_feature_names())
    print(date.toarray())

    return None


def tfidfvec():
    tf = TfidfVectorizer()
    r1, r2, r3 = cut()
    date = tf.fit_transform([
        r1, r2, r3
    ])

    print(tf.get_feature_names())
    print(date.toarray())

    return None


def mm():
    # 归一化
    mm = MinMaxScaler()

    data = mm.fit_transform([
        [90, 2, 10, 40],
        [60, 4, 15, 45],
        [75, 3, 13, 46]
    ])

    print(data)


def ss():
    # 标准化
    ss = StandardScaler()

    data = ss.fit_transform([
        [90, 2, 10, 40],
        [60, 4, 15, 45],
        [75, 3, 13, 46]
    ])
    print(data)

    return None


def imp():
    # 缺失值处理
    im = Imputer(missing_values="NaN", strategy="mean", axis=0)
    data = im.fit_transform([
        [90, 2, 10, 40],
        [np.nan, 4, 15, 45],
        [75, np.nan, 13, 46]
    ])
    print(data)

    return None


def var():
    #     删除底方差的特征
    var = VarianceThreshold(threshold=0.0)
    data = var.fit_transform([[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]])
    print(data)

    return None


if __name__ == "__main__":
    tfidfvec()
