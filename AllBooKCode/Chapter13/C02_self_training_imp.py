"""
文件名: AllBooKCode/Chapter12/C02_self_training_imp.py
创建时间: 2022/6/11  上午 星期六
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

import logging
from copy import deepcopy
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import sys
import numpy as np

sys.path.append('../')
from Chapter10.C16_svm_impl import SVM


class MySVM(SVM):
    def __init__(self, **kwargs):
        super(MySVM, self).__init__(**kwargs)

    def predict_proba(self, X):
        _, prob = self.predict(X, return_prob=True)
        return prob


class SelfTrainingClassifier(object):
    def __init__(self,
                 base_estimator,
                 threshold=0.75,
                 max_iter=10):
        self.base_estimator_ = base_estimator
        self.threshold = threshold
        self.max_iter = max_iter

    def fit(self, X, y):
        """
        拟合模型
        :param X:
        :param y:
        :return:
        """
        if not (0 <= self.threshold < 1):
            raise ValueError("参数 threshold 必须属于 [0,1) 的范围"
                             f"，当前传入为 {self.threshold}")

        has_label = y != -1  # -1（代表没有label） 的是False 非-1（有label）的是True，
        # 有label的地方为True，无label的地方为False

        if np.all(has_label):  # 如果全都有标签，则表示为有监督学习
            logging.warning("y 中均为有标签的数据样本")

        self.transduction_ = np.copy(y)  #
        self.labeled_iter_ = np.full_like(y, -1)  # 用来记录每个无标签样本被拟合的次数
        self.labeled_iter_[has_label] = 0
        self.n_iter_ = 0

        if not hasattr(self.base_estimator_, "predict_proba"):
            msg = "base_estimator ({}) 需要实现 predict_proba 方法来返回每个样本的预测概率!"
            raise ValueError(msg.format(type(self.base_estimator_).__name__))

        while not np.all(has_label) and (self.max_iter is None or
                                         self.n_iter_ < self.max_iter):
            # 如果(存在标签为-1的情况)且(最大迭代次数为空 或 当前迭代次数小于最大次数）
            self.n_iter_ += 1  # 累计迭代次数
            self.base_estimator_.fit(X[has_label],  # 取有标签的部分进行模型拟合
                                     self.transduction_[has_label])
            # ①先对每个没有标签的样本进行预测
            prob = self.base_estimator_.predict_proba(X[~has_label])  # 取没有标签的样本进行预测, [n_samples, n_classes]
            # ②以最大概率（这个最大概率可能没有超过设定的阈值）给一个初始的预测结果
            pred = self.base_estimator_.classes_[np.argmax(prob, axis=1)]  # 根据概率取预测标签 [n_samples,]
            max_proba = np.max(prob, axis=1)  # 取每个样本类别预测中的最大概率 [n_samples,]
            # ③选择预测概率超过阈值的样本索引
            selected = max_proba > self.threshold  # 通过阈值来进行确定哪些标签的预测结果是可信的
            selected_full = np.nonzero(~has_label)[0][selected]
            # ④将样本预测概率最大值对应的标签且概率同时大于阈值的预测结果更新到标签结果中
            self.transduction_[selected_full] = pred[selected]  # 更新标签
            has_label[selected_full] = True  # 设定原本无标签的对应样本为有标签状态
            self.labeled_iter_[selected_full] = self.n_iter_
            # labeled_iter_记录样本被拟合的次数，因为存在概率小于阈值的情况，所以可能需要拟合多次

            if selected_full.shape[0] == 0:
                # no changed labels
                self.termination_condition_ = "no_change"  # 结束标志：没有变化
                break

            print(f"第 {self.n_iter_} 次迭代结束后,"
                  f"有 {selected_full.shape[0]} 个未标记样本增加了新标签.")

        if self.n_iter_ == self.max_iter:
            self.termination_condition_ = "max_iter"  # 结束标志：达到最大迭代次数
        if np.all(has_label):
            self.termination_condition_ = "all_labeled"  # # 结束标志：所有样本都有了标记

        # 最后利用所有有标签的样本进行一次模型的拟合
        self.base_estimator_.fit(X[has_label], self.transduction_[has_label])
        self.classes_ = self.base_estimator_.classes_
        return self

    def predict(self, X):
        """
        预测部分实现
        :param X:  [n_samples, n_features]
        :return: y: ndarray, shape [n_samples,]
        """
        return self.base_estimator_.predict(X)

    def predict_proba(self, X):
        """
        预测概率部分实现
        :param X:  [n_samples, n_features]
        :return: y: ndarray, shape [n_samples,n_classes]
        """
        return self.base_estimator_.predict_proba(X)


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
    svm = MySVM()
    # svm = SVC(probability=True)
    print(f"一共有{np.sum(y_mixed == -1)}个样本无标签.")
    model = SelfTrainingClassifier(svm, threshold=0.6)
    model.fit(x_train, y_mixed)

    y_pred = model.predict(x_train)
    print(f"模型训练结束的标志为：{model.termination_condition_}")
    print(f"模型在训练集上的准确率为: {accuracy_score(y_pred, y_train)}")

    y_pred = model.predict(x_test)
    print(f"模型在测试集上的准确率为: {accuracy_score(y_pred, y_test)}")


if __name__ == '__main__':
    test_self_training()
    # 一共有27个样本无标签.
    # 第 1 次迭代结束后,有 14 个未标记样本增加了新标签.
    # 模型训练结束的标志为：no_change
    # 模型在训练集上的准确率为: 0.9904761904761905
    # 模型在测试集上的准确率为: 0.9777777777777777
