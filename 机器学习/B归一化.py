from sklearn.datasets import load_iris, fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

li = load_iris()

print("获取特征值")
print(li.data)

print("获取目标值")
print(li.target)

print("获取描述信息")
print(li.DESCR)

# 注意返回值，训练集 测试集
xtr, xte, ytr, yte = train_test_split(li.data, li.target, test_size=0.25)
print("训练特征值和目标值：", xtr, ytr)
print("测试特征值和目标值：", xte, yte)

news = fetch_20newsgroups(subset="all")
print("获取特征值\n")
# print(news.data)

print("获取目标值")
print(news.target)

print("获取描述信息")
print(news.DESCR)
