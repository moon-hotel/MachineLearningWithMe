"""
文件名: C21_manual_gradient_boost_reg.py
创建时间: 2022/10/30 20:32 下午
作者: @空字符
公众号: @月来客栈
知乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def make_data(n=20, visualization=False):
    """
    制作数据集
    :param n:
    :param visualization:
    :return:
    """
    np.random.seed(1000)
    x = np.linspace(1, 5, n)
    noise = np.random.randn(n)
    y_noise = 2 * x + 5. + noise * 2
    y = 2 * x + 5.
    if visualization:
        plt.scatter(x, y_noise, c='black')
        plt.plot(x, y)
        plt.show()
    return x.reshape(-1, 1), y_noise


def objective_function(y, y_hat):
    return 0.5 * (y - y_hat) ** 2


def negative_gradient(y, y_hat):
    """
    J 关于 y_hat 的负梯度
    由于后续需要用每个样本的梯度来更新预测结果
    所以这里并不用计算整体的平均梯度
    -(-(y- y_hat))
    :param y:
    :param y_hat:
    :return:
    """
    return (y - y_hat)


class MyGradientBoostRegression(object):
    """
    Gradient Boosting for Regression
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
        self.estimators_ = []
        self.loss_ = []

    def fit(self, X, y):
        """

        :param X: shape: [n_samples, n_features]
        :param y: shape: [n_samples,]
        :return:
        """
        self.raw_predictions = np.zeros(X.shape[0])  # [n_samples,]
        self._fit_stages(X, y, self.raw_predictions)

    def _fit_stages(self, X, y, raw_predictions):
        for i in range(self.n_estimators):
            grad = negative_gradient(y, raw_predictions)
            model = self.base_estimator()
            model.fit(X, grad)  # 这里其实是用于拟合梯度，因为在预测时无法计算得到真实梯度
            grad = model.predict(X)  # 当然，这里的grad也可以直接使用上面的真实值
            raw_predictions += self.learning_rate * grad  # 梯度下降更新预测结果
            self.loss_.append(np.sum(objective_function(y, raw_predictions)))
            self.estimators_.append(model)

    def predict(self, X):
        raw_predictions = np.zeros(X.shape[0])  # [n_samples,]
        for model in self.estimators_:
            grad = model.predict(X)  # 预测每个样本在当前boost序列中对应的梯度值
            raw_predictions += self.learning_rate * grad  # 梯度下降更新预测结果
        return raw_predictions


if __name__ == '__main__':
    x, y = make_data(200, False)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    n_estimators = 100
    my_boost = MyGradientBoostRegression(n_estimators=n_estimators,
                                         base_estimator=LinearRegression,
                                         learning_rate=0.1)
    my_boost.fit(x_train, y_train)
    plt.plot(range(n_estimators), my_boost.loss_)
    plt.xlabel("boosting steps")
    plt.ylabel("loss on training data")
    plt.show()
    loss = np.sum(objective_function(y_test, my_boost.predict(x_test)))
    print(f"loss in test by my_boost {loss}")

    model = LinearRegression()
    model.fit(x_train, y_train)
    loss = np.sum(objective_function(y_test, model.predict(x_test)))
    print(f"loss in test by LinearRegression {loss}")
