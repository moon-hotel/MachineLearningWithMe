"""
文件名: AllBooKCode/Chapter11/C02_PCA_train.py
创建时间: 2022/6/5
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


def load_data(reduction=False, n_components=3):
    x, y = load_iris(return_X_y=True)
    x_train, x_test, y_train, y_test = \
        train_test_split(x, y, test_size=0.3, random_state=2022)
    if reduction:
        pca = PCA(n_components)
        x_train = pca.fit_transform(x_train)
        x_test = pca.transform(x_test)
    return x_train, x_test, y_train, y_test


def decision_tree():
    x_train, x_test, y_train, y_test = load_data(reduction=False)
    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)
    print(f"降维前的准确率为：{acc}，形状为{x_train.shape}")

    x_train, x_test, y_train, y_test = load_data(reduction=True)
    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)
    print(f"降维后的准确率为：{acc}，形状为{x_train.shape}")


if __name__ == '__main__':
    decision_tree()
    # 降维前的准确率为：0.9555555555555556，形状为(105, 4)
    # 降维后的准确率为：0.9777777777777777，形状为(105, 3)
