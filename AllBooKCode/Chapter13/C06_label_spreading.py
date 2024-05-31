"""
文件名: AllBooKCode/Chapter12/C06_label_spreading.py
创建时间:
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

from copy import deepcopy
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import LabelSpreading


def load_data():
    x, y = load_iris(return_X_y=True)
    x_train, x_test, y_train, y_test = \
        train_test_split(x, y, test_size=0.3, random_state=2022)
    rng = np.random.RandomState(20)
    random_unlabeled_points = rng.rand(y_train.shape[0]) < 0.8
    y_mixed = deepcopy(y_train)
    y_mixed[random_unlabeled_points] = -1
    return x_train, x_test, y_train, y_test, y_mixed


def test_label_spreding():
    x_train, x_test, y_train, y_test, y_mixed = load_data()
    ls = LabelSpreading()
    ls.fit(x_train, y_mixed)
    print("Label Spreading")
    print(f"训练集上的准确率为：{ls.score(x_train, y_train)}")
    print(f"测试集上的准确率为：{ls.score(x_test, y_test)}")


if __name__ == '__main__':
    test_label_spreding()

    # Label Spreading
    # 训练集上的准确率为：0.9714285714285714
    # 测试集上的准确率为：1.0
