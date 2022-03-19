from sklearn.naive_bayes import GaussianNB
import numpy as np
import logging
import sys


def load_simple_data():
    import numpy as np
    x = np.array([[0.8, 0.2, 0.3, 0.5, 0.1, 0.7, 0.1, 0.3, 0.6, 0.6],
                  [0.3, 0.8, 0.4, 0.5, 0.6, 0.5, 0.6, 0.7, 0.1, 0.6],
                  [0.3, 0.3, 0.6, 0.7, 0.8, 0.8, 0.8, 0.3, 0.3, 0.6]]).transpose()
    y = np.array([1, 1, 2, 2, 2, 1, 1, 0, 1, 0])
    return x, y


class MyGaussianNB(object):
    """
    Gaussian Naive Bayes 实现
    """

    def __init__(self, ):
        pass

    def _init_counters(self, X, y):
        self.classes_ = np.sort(np.unique(y))  # 排序是为了后面依次遍历每个类别
        n_features = X.shape[1]
        n_classes = len(self.classes_)
        self.mu_ = np.zeros((n_classes, n_features))  # mu_[i][j]表示第i个类别的第j个特征对应的期望
        self.sigma2_ = np.zeros((n_classes, n_features))  # sigma2_[i][j]表示第i个类别的第j个特征对应的方差
        self.class_count_ = np.zeros(n_classes, dtype=np.float64)
        # 统计每个类别下的样本数
        self.class_prior_ = np.zeros(len(self.classes_), dtype=np.float64)
        # 初始化先验概率

    def fit(self, X, y):
        """implementation of Gaussian NB fitting.

        Parameters
        ----------
        X : array-like of shape [n_samples, n_features]
        y : array-like of shape [n_samples,]
            Target values.
        """
        self._init_counters(X, y)
        for i, y_i in enumerate(self.classes_):  # 遍历每一个类别
            X_i = X[y == y_i, :]  # 取类别y_i对应的所有样本
            self.mu_[i, :] = np.mean(X_i, axis=0)  # 计算期望
            self.sigma2_[i, :] = np.var(X_i, axis=0)  # 计算方差
            self.class_count_[i] += X_i.shape[0]  # 类别y_i对应的样本数量
        self.class_prior_ = self.class_count_ / self.class_count_.sum()
        logging.debug(f"\n期望mu = {self.mu_}")
        logging.debug(f"\n方差sigma = {self.sigma2_}")
        logging.debug(f"\n先验概率 = {self.class_prior_}")
        logging.debug(f"\nlog先验概率 = {np.log(self.class_prior_)}")
        return self

    def _joint_likelihood(self, X):
        """
        预测是计算联合概率
        :param X: shape: [n_samples,n_features]
        :return:
        """
        joint_likelihood = []
        for i in range(np.size(self.classes_)):
            jointi = np.log(self.class_prior_[i])  # shape: [1,]
            n_ij = - 0.5 * np.sum(np.log(2. * np.pi * self.sigma2_[i, :]))
            n_ij -= 0.5 * np.sum(((X - self.mu_[i, :]) ** 2) /
                                 (self.sigma2_[i, :]), 1)  # shape: [n_samples,]
            joint_likelihood.append(jointi + n_ij)  # [[n_samples,1],..[n_samples,1]]
        joint_likelihood = np.array(joint_likelihood).T  # [n_samples,n_classes]
        return joint_likelihood

    def predict(self, X, with_prob=False):
        """
        极大化后验概率进行预测
        Parameters
        ----------
        X : shape: [n_samples,n_features]
        """
        from scipy.special import softmax
        jll = self._joint_likelihood(X)
        logging.debug(f"\n样本预测原始概率为：{jll}")
        y_pred = self.classes_[np.argmax(jll, axis=1)]
        if with_prob:
            prob = softmax(jll)
            return y_pred, prob
        return y_pred


def test_naive_bayes():
    x, y = load_simple_data()
    logging.info(f"========== MyGaussianNB 运行结果 ==========")
    model = MyGaussianNB()
    model.fit(x, y)
    logging.info(f"预测结果: {model.predict(np.array([[0.5, 0.12, 0.218]]), with_prob=True)}")
    logging.info(f"========== GaussianNB 运行结果 ==========")
    model = GaussianNB()
    model.fit(x, y)
    logging.debug(f"\n期望mu = {model.theta_}")
    logging.debug(f"\n方差sigma = {model.sigma_}")
    logging.debug(f"\n先验概率 = {model.class_prior_}")
    logging.debug(f"\nlog先验概率 = {np.log(model.class_prior_)}")
    logging.info(f"预测结果: {model.predict(np.array([[0.5, 0.12, 0.218]]))}")


if __name__ == '__main__':
    formatter = '[%(asctime)s] - %(levelname)s: %(message)s'
    logging.basicConfig(level=logging.INFO,  # 如果需要查看简略信息可将该参数改为logging.INFO
                        format=formatter,  # 关于Logging模块的详细使用可参加文章https://www.ylkz.life/tools/p10958151/
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[logging.StreamHandler(sys.stdout)])
    test_naive_bayes()
