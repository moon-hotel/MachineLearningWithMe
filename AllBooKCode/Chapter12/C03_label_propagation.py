"""
文件名: AllBooKCode/Chapter12/C03_label_propagation.py
创建时间: 2022/6/22
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

from copy import deepcopy
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import LabelPropagation


def load_data():
    x, y = load_iris(return_X_y=True)
    x_train, x_test, y_train, y_test = \
        train_test_split(x, y, test_size=0.3, random_state=2022)
    rng = np.random.RandomState(20)
    random_unlabeled_points = rng.rand(y_train.shape[0]) < 0.8
    y_mixed = deepcopy(y_train)
    y_mixed[random_unlabeled_points] = -1
    return x_train, x_test, y_train, y_test, y_mixed


def test_label_propagation():
    x_train, x_test, y_train, y_test, y_mixed = load_data()
    model = LabelPropagation(gamma=20, max_iter=1000)
    model.fit(x_train, y_mixed)
    y_pred = model.predict(x_train)
    print(f"模型在训练集上的准确率为: {accuracy_score(y_pred, y_train)}")
    y_pred = model.predict(x_test)
    print(f"模型在测试集上的准确率为: {accuracy_score(y_pred, y_test)}")


if __name__ == '__main__':
    test_label_propagation()

    # 模型在训练集上的准确率为: 0.9619047619047619
    # 模型在测试集上的准确率为: 1.0
