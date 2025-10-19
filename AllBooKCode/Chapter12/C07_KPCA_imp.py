"""
文件名: AllBooKCode/Chapter11/C07_KPCA_imp.py
创建时间: 2022/8/16
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

import numpy as np
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.model_selection import train_test_split
from sklearn.decomposition import KernelPCA


class MyKernelPCA(object):
    """
    Parameters
    ----------
    n_components : int, default=None 指定主成分数量，默认None即保留所有主成分
    kernel: 指定核函数类型，可选有：'linear',  'poly','rbf', 'sigmoid' default='linear'
    gamma:  核函数 'poly','rbf', 'sigmoid' 中的核系数
    degree: 多项式核 'poly' 中的多项式次数
    coef0: 'poly', 'sigmoid' 中的系数
    """

    def __init__(self, n_components=None,
                 kernel='linear',
                 gamma=2.,
                 degree=3,
                 coef0=1.):
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0

    def _get_center_kernel(self, X, Y=None):
        """
        计算得到核矩阵，当Y为None时则是计算X中两两样本之间的结果
        :param X: shape: [n_samples,n_features]
        :param Y: shape: [n_samples,n_features]
        :return:
        """
        params = {"gamma": self.gamma,
                  "degree": self.degree,
                  "coef0": self.coef0}
        K = pairwise_kernels(X, Y, metric=self.kernel,
                             filter_params=True,
                             **params)  # [n_samples,n_samples]

        if Y is None:  # 方阵,训练集上的时候
            l_mat = np.ones(K.shape) / K.shape[0]
            KM = K - (K @ l_mat) - (l_mat @ K) + ((l_mat @ K) @ l_mat)
        else:  # 非方阵测试集上，  其实只用下面这种计算方式可以兼容两种情况
            row_mean = (np.sum(K, axis=1) / K.shape[0])[:, None]  # 行上求平均
            col_mean = np.sum(K, axis=0) / K.shape[0]  # 列上求平均
            all_mean = np.sum(row_mean) / K.shape[0]  # 所有值的平均
            KM = K - col_mean - row_mean + all_mean  # 这里会用到广播机制
        return KM

    def fit(self, X):
        """
        拟合模型
        :param X: shape: [n_samples,n_features]
        :return:
        """
        self._X = X
        self.KM = self._get_center_kernel(X)  # self.KM: [n_samples,n_samples]
        # 计算特征值和特征向量
        w, v = np.linalg.eigh(self.KM)
        # v[:,i] 是特征值w[i]所对应的特征向量
        idx = np.argsort(w)[::-1]  # 获取特征值降序排序的索引
        self.lambdas_ = w[idx]  # [k,]  进行降序排列
        self.alphas_ = v[:, idx][:, :self.n_components]  # [n_samples,n_components]，  排序
        return self

    def fit_transform(self, X):
        """
        :param X: 训练样本 shape: [n_samples,n_features]
        :return:
        """
        self.fit(X)
        # [n_samples,n_samples] @ [n_samples,n_components]
        return np.matmul(self.KM, self.alphas_)

    def transform(self, X):
        """
        :param X: 新样本 shape:[n_samples,n_features]
        :return:
        """
        # 降维
        KM = self._get_center_kernel(X, self._X)
        print(KM.shape)
        return np.matmul(KM, self.alphas_)


def make_nonlinear_cla_data():
    num_points = 1000
    x, y = make_circles(n_samples=num_points, factor=0.2, noise=0.1,
                        random_state=np.random.seed(10))
    x_train, x_test, y_train, y_test = \
        train_test_split(x, y, test_size=0.3, random_state=2020)
    return x_train, x_test, y_train, y_test


def test_MyKernelPCA():
    x_train, x_test, y_train, y_test = make_nonlinear_cla_data()
    my_kernel_pca = MyKernelPCA(n_components=2, kernel='rbf', gamma=11.)
    x_d = my_kernel_pca.fit_transform(x_train)
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train)
    plt.subplot(1, 2, 2)
    plt.scatter(x_d[:, 0], x_d[:, 1], c=y_train)
    plt.tight_layout()
    plt.show()


def comp():
    x_train, x_test, y_train, y_test = make_nonlinear_cla_data()

    my_kernel_pca = MyKernelPCA(n_components=2, kernel='rbf', gamma=11.)
    my_kernel_pca.fit(x_train)
    x_d = my_kernel_pca.transform(x_test)
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title("MyKernelPCA on test data")
    plt.scatter(x_d[:, 0], x_d[:, 1], c=y_test)
    plt.tick_params(axis='x', labelsize=15)  # x轴刻度数字大小
    plt.tick_params(axis='y', labelsize=15)  # y轴刻度数字大小

    kernel_pca = KernelPCA(n_components=2, kernel='rbf', gamma=11.)
    kernel_pca.fit(x_train)
    x_d = kernel_pca.transform(x_test)
    plt.subplot(1, 2, 2)
    plt.title("KernelPCA on test data")
    plt.scatter(x_d[:, 0], x_d[:, 1], c=y_test)
    plt.tick_params(axis='x', labelsize=15)  # x轴刻度数字大小
    plt.tick_params(axis='y', labelsize=15)  # y轴刻度数字大小
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    test_MyKernelPCA()
    comp()
