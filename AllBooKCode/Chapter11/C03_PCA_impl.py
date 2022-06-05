from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler


class MyPCA(object):
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        self.standar_scaler = StandardScaler()
        X = self.standar_scaler.fit_transform(X)
        # 计算特征值和特征向量
        w, v = np.linalg.eig(np.matmul(X.T, X) / len(X))
        # v[:,i] 是特征值w[i]所对应的特征向量
        idx = np.argsort(w)[::-1]  # 获取特征值降序排序的索引
        self.w = w[idx]  # [k,]  进行降序排列
        self.v = v[:, idx][:, :self.n_components]  # [n,k]，  排序
        return self

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def transform(self, X):
        # 降维
        X = self.standar_scaler.transform(X)
        return np.matmul(X, self.v)  # [m,n] @ [n,k] = [m,k]


def load_data(reduction_method='my_pca', n_components=3):
    x, y = load_iris(return_X_y=True)
    x_train, x_test, y_train, y_test = \
        train_test_split(x, y, test_size=0.3, random_state=2022)
    if reduction_method == "my_pca":
        pca = MyPCA(n_components)
    else:
        pca = PCA(n_components)
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)
    return x_train, x_test, y_train, y_test


def test_PCA():
    x_train, x_test, y_train, y_test = load_data(reduction_method="my_pca",
                                                 n_components=3)

    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)
    print(f"MyPCA降维后的准确率为：{acc}，形状为{x_train.shape}")

    x_train, x_test, y_train, y_test = load_data(reduction_method="sklearn",
                                                 n_components=3)
    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)
    print(f"PCA降维后的准确率为：{acc}，形状为{x_train.shape}")


if __name__ == '__main__':
    test_PCA()
