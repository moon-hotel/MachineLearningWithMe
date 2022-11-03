"""
文件名: C22_manual_gradient_boost_cla.py
创建时间: 2022/11/1 7:41 下午
作者: @空字符
公众号: @月来客栈
知乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from scipy.special import logsumexp
from sklearn.datasets import load_iris


def negative_gradient(y, raw_predictions, k=0):
    """
    损失函数关于类别k对应的梯度
    :param y: shape: [n_samples,]
    :param raw_predictions: 预测得到的概率值 shape: [n_samples,n_classes]
    :param k: [n_samples,]
    :return:
    """
    y = np.array(y == k, dtype=np.float64)
    a = y - np.nan_to_num(np.exp(raw_predictions[:, k] - logsumexp(raw_predictions, axis=1)))
    return a


def objective_function(y, raw_predictions, K):
    """
    计算损失 Multinomial deviance loss function for multi-class classification.
    L(y,p(x))=-\sum_{i=1}^m\left(\sum_{k=1}^K\mathbb{I}(y_i=c_k)f_k(x)+
                    \log\left(\sum_{l=1}^Ke^{f_l(x)}\right)\right)
    :param y: [n_samples,]
    :param raw_predictions: 预测得到的概率值 shape: [n_samples,n_classes]
    :param K:
    :return:
    """
    Y = np.zeros((y.shape[0], K), dtype=np.float64)
    for k in range(K):
        Y[:, k] = y == k
    return np.average(-1 * (Y * raw_predictions).sum(axis=1)
                      + logsumexp(raw_predictions, axis=1))


class MyGradientBoostClassifier(object):
    """
    Gradient Boosting for Classification
    author: 公众号：@月来客栈
            知乎：https://www.zhihu.com/people/the_lastest
    Parameters:
        learning_rate: 学习率
        n_estimators: boosting 次数
        base_estimator: 预测梯度时使用到的回归模型
    """

    def __init__(self,
                 learning_rate=0.1,
                 n_estimators=100,
                 base_estimator=None):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator
        self.loss_ = []

    def fit(self, X, y):
        """
        拟合模型
        :param X: shape: [n_samples, n_features]
        :param y: shape: [n_samples,]
        :return:
        """
        self.classes_, y = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        self.estimators_ = np.empty((self.n_estimators, self.n_classes_), dtype=object)
        self.raw_predictions = np.zeros([X.shape[0], self.n_classes_])  # [n_samples,n_classes]
        self._fit_stages(X, y, self.raw_predictions)

    def _fit_stages(self, X, y, raw_predictions):
        """
        拟合
        :param X: shape [n_samples, n_features]
        :param y: shape [n_samples,]
        :param raw_predictions:  [n_samples,n_classes] 概率值
        :return:
        """
        for i in range(self.n_estimators):
            for k in range(self.n_classes_):
                grad = negative_gradient(y, raw_predictions, k)  # 在训练集上计算真实梯度
                model = self.base_estimator()
                model.fit(X, grad)  # 这里其实是用于拟合梯度，因为在预测时无法计算得到真实梯度
                # grad = model.predict(X)  # 当然，这里的grad也可以使用模型的预测实值
                raw_predictions[:, k] += self.learning_rate * grad  # 梯度下降更新预测概率
                self.estimators_[i, k] = model  # 保存每一次boosting对应第k个类别的模型
            self.loss_.append(objective_function(y, raw_predictions, self.n_classes_))

    def predict_prob(self, X):
        """
        预测概率
        :param X: shape: [n_samples, n_features]
        :return:
        """
        raw_predictions = np.zeros([X.shape[0], self.n_classes_])  # 初始化概率值
        for i in range(self.n_estimators):  # 遍历每一次boosting每个类别对应的梯度预测模型
            for k in range(self.n_classes_):
                model = self.estimators_[i, k]
                grad = model.predict(X)  # 预测每个样本在当前boost序列中对应的梯度值
                raw_predictions[:, k] += self.learning_rate * grad  # 梯度下降更新预测结果
        return raw_predictions

    def predict(self, X):
        """
        返回预测标签
        :param X: [n_samples, n_features]
        :return: [n_samples,]
        """
        raw_predictions = self.predict_prob(X)
        return np.argmax(raw_predictions, axis=1)


if __name__ == '__main__':
    x, y = load_iris(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    my_boost = MyGradientBoostClassifier(base_estimator=DecisionTreeRegressor,
                                         learning_rate=0.3)
    my_boost.fit(x_train, y_train)
    y_hat = my_boost.predict(x_train)
    print(f"MyGradientBoostClassifier accuracy on training data {accuracy_score(y_train, y_hat)}")
    y_hat = my_boost.predict(x_test)
    print(f"MyGradientBoostClassifier accuracy on testing data {accuracy_score(y_test, y_hat)}")

    plt.plot(range(my_boost.n_estimators), my_boost.loss_)
    plt.xlabel("boosting steps")
    plt.ylabel("loss on training data")
    plt.show()

    print("===========")
    boost = GradientBoostingClassifier(init='zero')
    boost.fit(x_train, y_train)
    print(f"GradientBoostingClassifier acc on training data: {boost.score(x_train, y_train)}")
    print(f"GradientBoostingClassifier acc on testing data: {boost.score(x_test, y_test)}")
