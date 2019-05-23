from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd


def knncls():
    """
    K 近邻预测用户签到位置
    :return:
    """
    # 1 读取数据
    data = pd.read_csv("./data/train.csv")

    # 2 处理数据
    # 缩小数据范围
    data = data.query("x > 1.0 & x < 1.25 & y > 2.5 & y < 2.75 ")

    # 时间处理
    time_value = pd.to_datetime(data['time'], unit='s')

    # 把日期格式转换为字典格式
    time_value = pd.DatetimeIndex(time_value)

    # 构造特征
    # data['day'] = time_value.day
    # data['hour'] = time_value.hour
    data['weekday'] = time_value.weekday

    # 删除时间戳特征 1：列
    data = data.drop(['time'], axis=1)

    # 把签到数量少于n个目标位置删除
    place_count = data.groupby("place_id").count()
    tf = place_count[place_count.row_id > 3].reset_index()
    data = data[data['place_id'].isin(tf.place_id)]

    data = data.drop(['row_id'], axis=1)

    # 取出数据中的目标值和特征值
    y = data['place_id']
    x = data.drop(['place_id'], axis=1)

    # 进行数据的分割 训练集 测试机
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # 3 特征工程（标准化）
    std = StandardScaler()
    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)

    # 4 进行算法流程
    knn = KNeighborsClassifier(n_neighbors=8)
    knn.fit(x_train, y_train)

    # y_predict = knn.predict(x_test)
    # print(y_predict)

    # 得出准确率
    print("准确率：", knn.score(x_test, y_test))

    return None


if __name__ == "__main__":
    knncls()
