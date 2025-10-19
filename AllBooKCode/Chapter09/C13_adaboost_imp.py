import numpy as np
import sys

sys.path.append('../')
from Chapter07.C02_naive_bayes_multinomial import MyMultinomialNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from copy import deepcopy
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_iris

np.random.seed(1009)  # 固定结果


class MyAdaBoostClassifier():
    """
    多分类AdaBoost实现
    """

    def __init__(self,
                 base_estimator=None,
                 n_estimators=50,
                 algorithm='SAMME'):
        self.algorithm = algorithm
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator
        self.estimators_ = []  # 保存所有的分类器
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        # 保存每个模型对应的权重，
        self.estimator_errors_ = np.zeros(self.n_estimators, dtype=np.float64)
        # 保存每个模型拟合后对应的误差

    def _single_boost_samme(self, iboost, X, y, sample_weight):
        estimator = deepcopy(self.base_estimator)  # 克隆一个分类器
        self.estimators_.append(estimator)  # 保存每个分类器
        estimator.fit(X, y, sample_weight=sample_weight)
        y_predict = estimator.predict(X)
        if iboost == 0:
            self.classes_ = getattr(estimator, 'classes_', None)
            self.n_classes_ = len(self.classes_)

        incorrect = y_predict != y
        estimator_error = np.average(incorrect, weights=sample_weight, axis=0)
        # estimator_error = sum(incorrect * sample_weight) / sum(sample_weight) # 同上

        # Stop if the error is at least as bad as random guessing
        # 当不满足条件时则不再对后续分类器进行训练
        if estimator_error >= 1. - (1. / self.n_classes_):
            self.estimators_.pop(-1)
            if len(self.estimators_) == 0:
                raise ValueError('BaseClassifier in AdaBoostClassifier '
                                 'ensemble is worse than random, ensemble '
                                 'can not be fit.')
            return None, None, None

        estimator_weight = np.log((1. - estimator_error) /
                                  estimator_error) + np.log(self.n_classes_ - 1.)
        sample_weight *= np.exp(estimator_weight * incorrect)
        return sample_weight, estimator_weight, estimator_error

    def _boost(self, iboost, X, y, sample_weight):
        if self.algorithm == 'SAMME':
            return self._single_boost_samme(iboost, X, y, sample_weight)

    def fit(self, X, y, sample_weight=None):
        if sample_weight is None:
            sample_weight = np.ones(len(X), dtype=np.float64)  # 初始化权重全为1
        sample_weight /= sample_weight.sum()  # 标准化权重

        for iboost in range(self.n_estimators):
            sample_weight, estimator_weight, estimator_error = self._boost(
                iboost, X, y, sample_weight)
            # Early termination
            if sample_weight is None:
                break
            self.estimator_weights_[iboost] = estimator_weight
            self.estimator_errors_[iboost] = estimator_error

            if estimator_error == 0:
                break
            sample_weight_sum = np.sum(sample_weight)
            if sample_weight_sum <= 0:
                break
            sample_weight /= sample_weight_sum  # 标准化样本权重

        return self

    def predict(self, X):
        """
        C(x)=\arg \max_{k}=\sum_{m=1}^M\alpha^{(m)}\cdot \mathbb{I}(g^{(m)}(x)=k)
        :param X: shape: [n_samples,n_features]
        :return:
        """
        classes = self.classes_[:, np.newaxis]  # [n_classes,1]
        pred = np.zeros([len(X), len(self.classes_)])  # [n_samples,n_classes]
        for estimator, alpha in zip(self.estimators_, self.estimator_weights_):
            correct = estimator.predict(X) == classes  # [n_classes,n_samples]
            result = alpha * correct.T  # [n_samples,n_classes]
            pred += result  # [n_samples,n_classes] 把每个分类器的结果相加
        pred /= self.estimator_weights_.sum()
        y_pred = np.argmax(pred, axis=1)
        return y_pred


if __name__ == '__main__':
    x, y = load_iris(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    clf = MyAdaBoostClassifier(base_estimator=MyMultinomialNB(), n_estimators=10)
    clf.fit(x_train, y_train)
    print(accuracy_score(y_test, clf.predict(x_test)))
    # print(clf.estimator_weights_)
    # print(clf.estimator_errors_)

    clf = AdaBoostClassifier(estimator=MultinomialNB(), n_estimators=10)
    clf.fit(x_train, y_train)
    print(accuracy_score(y_test, clf.predict(x_test)))
    # print(clf.estimator_weights_)
    # print(clf.estimator_errors_)
