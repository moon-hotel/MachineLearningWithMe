from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KDTree
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import numpy as np


def load_data():
    x, y = load_iris(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=0.3, random_state=20)
    return x_train, x_test, y_train, y_test


def impl_by_sklearn(x_train, x_test, y_train, y_test, k=5):
    model = KNeighborsClassifier(n_neighbors=k, leaf_size=30)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print("impl_by_sklearn 准确率：", accuracy_score(y_test, y_pred))


def impl_by_ours(x_train, x_test, y_train, y_test, k):
    kd_tree = KDTree(x_train, leaf_size=30)  # 构建一颗KD树
    dist, ind = kd_tree.query(x_test, k=k)  # 寻找离x_test最近的k个点，返回距离和索引
    # dist: shape (n,k) dist[i] 表示离第i个样本点最近的k个样本点的距离，dist[i][0] = 0
    # ind: shape (n,k) ind[i] 表示离第i个样本点最近的k个样本点的索引，ind[i][0]是第i个样本点自己本身
    query_label = y_train[ind][:, 1:]
    y_pred = get_pred_labels(query_label)
    print("impl_by_ours 准确率：", accuracy_score(y_test, y_pred))


def get_pred_labels(query_label):
    """
    根据query_label返回每个样本对应的标签
    :param query_label: 二维数组， query_label[i] 表示离第i个样本最近的k-1个样本点对应的正确标签
    :return:
    """
    y_pred = [0] * len(query_label)
    for i, label in enumerate(query_label):
        max_freq = 0
        count_dict = {}
        for l in label:
            count_dict[l] = count_dict.setdefault(l, 0) + 1
            if count_dict[l] > max_freq:
                max_freq = count_dict[l]
                y_pred[i] = l
    return np.array(y_pred)




if __name__ == '__main__':
    x_train, x_test, y_train, y_test = load_data()
    k = 5
    impl_by_sklearn(x_train, x_test, y_train, y_test, k)
    impl_by_ours(x_train, x_test, y_train, y_test, k)
