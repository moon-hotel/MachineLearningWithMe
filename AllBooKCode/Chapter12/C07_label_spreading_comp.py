"""
文件名: AllBooKCode/Chapter12/C07_label_spreading_comp.py
创建时间:
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

from copy import deepcopy
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelSpreading
from C04_label_propagation_imp import LabelPropagation
from sklearn.datasets import load_digits


def load_data(noise_rate=0.1):
    n_class = 10
    x, y = load_digits(n_class=n_class, return_X_y=True)
    # 划分30%为测试集，70%为训练集
    x_train, x_test, y_train, y_test = \
        train_test_split(x, y, test_size=0.3, random_state=2022)
    rng = np.random.RandomState(20)
    # 在训练集中将其中80%样本的标签去掉，置为-1
    random_unlabeled_points = rng.rand(y_train.shape[0]) < 0.8
    y_mixed = deepcopy(y_train)
    y_mixed[random_unlabeled_points] = -1
    for i in range(len(y_mixed)):
        if y_mixed[i] == -1:
            continue
        if rng.random() < noise_rate:  # 在训练集中，将有标签的样本中的noise_rate替换为随机标签
            candidate_ids = np.random.permutation(n_class).tolist()
            candidate_ids.remove(y_mixed[i])
            y_mixed[i] = candidate_ids[0]
    ss = StandardScaler()
    x_train = ss.fit_transform(x_train)
    x_test = ss.transform(x_test)
    return x_train, x_test, y_train, y_test, y_mixed


def comp(p=0.1):
    x_train, x_test, y_train, y_test, y_mixed = load_data(noise_rate=p)

    lp = LabelPropagation(max_iter=200)
    lp.fit(x_train, y_mixed)
    print("LabelPropagation")
    print(f"训练集上的准确率为：{lp.score(x_train, y_train)}")
    print(f"测试集上的准确率为：{lp.score(x_test, y_test)}")

    ls = LabelSpreading(max_iter=200)
    paras = {'alpha': np.arange(0.01, 1, 0.02)}
    gs = GridSearchCV(ls, paras, verbose=1, cv=3)
    gs.fit(x_train, y_mixed)
    print("LabelSpreading")
    print('最佳模型:', gs.best_params_)
    print(f"训练集上的准确率为：{gs.score(x_train, y_train)}")
    print(f"测试集上的准确率为：{gs.score(x_test, y_test)}")


if __name__ == '__main__':
    comp()
