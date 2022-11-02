from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from scipy.special import logsumexp
from sklearn.datasets import load_iris


def negative_gradient(y, y_hat, k=0):
    a = y - np.nan_to_num(np.exp(y_hat[:, k] - logsumexp(y_hat, axis=1)))
    return a


class MyGradientBoostClassifier(object):
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

        :param X: shape: [n_samples, n_features]
        :param y: shape: [n_samples,]
        :return:
        """
        self.n_classes_ = len(np.unique(y))
        self.estimators_ = np.empty((self.n_estimators, self.n_classes_), dtype=object)
        self.raw_predictions = np.zeros([X.shape[0], self.n_classes_])  # [n_samples,n_classes]
        self.classes_, y = np.unique(y, return_inverse=True)
        self._fit_stages(X, y, self.raw_predictions)

    def _fit_stages(self, X, y, raw_predictions):
        X = np.array(X, dtype=np.float32)
        for i in range(self.n_estimators):
            for k in range(self.n_classes_):
                grad = negative_gradient(y, raw_predictions, k)
                model = self.base_estimator(splitter='best')
                model.fit(X, grad)  # 这里其实是用于拟合梯度，因为在预测时无法计算得到真实梯度
                # out[:, k] += scale * tree.predict(X).ravel()
                grad = model.predict(X)  # 当然，这里的grad也可以直接使用上面的真实值
                raw_predictions[:, k] += self.learning_rate * grad  # 梯度下降更新预测结果
                self.estimators_[i, k] = model

    def predict(self, X):
        raw_predictions = np.zeros([X.shape[0], self.n_classes_])
        for i in range(self.n_estimators):
            for k in range(self.n_classes_):
                model = self.estimators_[i, k]
                grad = model.predict(X)  # 预测每个样本在当前boost序列中对应的梯度值
                raw_predictions[:, k] += self.learning_rate * grad  # 梯度下降更新预测结果
        return np.argmax(raw_predictions, axis=1)


if __name__ == '__main__':
    x, y = load_iris(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    my_boost = MyGradientBoostClassifier(base_estimator=DecisionTreeRegressor,
                                         learning_rate=0.4)
    my_boost.fit(x_train, y_train)
    y_hat = my_boost.predict(x_train)
    print(accuracy_score(y_train, y_hat))
    y_hat = my_boost.predict(x_test)
    print(accuracy_score(y_test, y_hat))

    print("===========")
    my_boost = GradientBoostingClassifier(init='zero')
    my_boost.fit(x_train, y_train)
    print(my_boost.score(x_train, y_train))
    print(my_boost.score(x_test, y_test))


