"""
文件名: AllBooKCode/Chapter12/C01_self_training.py
创建时间: 2022/6/11  上午 星期六
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

from copy import deepcopy
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.svm import SVC
import numpy as np


def load_data():
    x, y = load_iris(return_X_y=True)
    x_train, x_test, y_train, y_test = \
        train_test_split(x, y, test_size=0.3, random_state=2022)
    rng = np.random.RandomState(20)
    random_unlabeled_points = rng.rand(y_train.shape[0]) < 0.3
    y_mixed = deepcopy(y_train)
    y_mixed[random_unlabeled_points] = -1
    return x_train, x_test, y_train, y_test, y_mixed


def test_self_training():
    x_train, x_test, y_train, y_test, y_mixed = load_data()
    svm = SVC(probability=True)
    print(f"一共有{np.sum(y_mixed == -1)}个样本无标签.")
    model = SelfTrainingClassifier(svm, threshold=0.6)
    model.fit(x_train, y_mixed)

    y_pred = model.predict(x_test)
    print(f"模型在测试集上的准确率为: {accuracy_score(y_pred, y_test)}")


if __name__ == '__main__':
    test_self_training()

    # 一共有27个样本无标签.
    # 模型在测试集上的准确率为: 0.9777777777777777
