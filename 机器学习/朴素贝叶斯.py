from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


def bayes():
    """
    朴素贝叶斯
    :return:
    """

    news = fetch_20newsgroups(subset="all")

    # 数据分割
    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25)

    # 对数据集进行特征抽取
    tf = TfidfVectorizer()

    # 以训练集中的词的列表进行每篇文章重要性统计
    x_train = tf.fit_transform(x_train)
    x_test = tf.transform(x_test)

    # 进行朴素贝叶斯算法
    mlt = MultinomialNB(alpha=1.0)
    mlt.fit(x_train, y_train)

    y_predict = mlt.predict(x_test)
    print(y_predict)
    print("准确率：", mlt.score(x_test, y_test))
    return None


if __name__ == "__main__":
    bayes()
