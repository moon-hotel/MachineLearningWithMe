from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np


def load_data(indices=False):
    data = load_iris()

    x, y = data.data, data.target
    if indices:
        x = x[:, 2:4]
    x_train, x_test, y_train, y_test = \
        train_test_split(x, y, test_size=0.3,
                         shuffle=True, random_state=20)
    return x_train, x_test, y_train, y_test


def rfc1(x_train, x_test, y_train, y_test):
    forest = RandomForestClassifier(n_estimators=10, n_jobs=-1, random_state=9)
    forest.fit(x_train, y_train)
    print("包含所有维度时的准确率：", forest.score(x_test, y_test))
    importances = forest.feature_importances_
    print('每个维度对应的重要性：', importances)
    indices = np.argsort(importances)[::-1]  # a[::-1]让a逆序输出
    print('按维度重要性排序的维度的序号：', indices)


def frc2(x_train, x_test, y_train, y_test):
    forest = RandomForestClassifier(n_estimators=10, n_jobs=-1, random_state=9)
    forest.fit(x_train, y_train)
    print("仅包含两个维度时的准确率：", forest.score(x_test, y_test))


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = load_data()
    rfc1(x_train, x_test, y_train, y_test)
    x_train, x_test, y_train, y_test = load_data(indices=True)
    frc2(x_train, x_test, y_train, y_test)
