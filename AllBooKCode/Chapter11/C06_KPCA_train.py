"""
文件名: AllBooKCode/Chapter11/C06_KPCA_train.py
创建时间: 2022/8/14
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt


def make_nonlinear_cla_data():
    num_points = 500
    x, y = make_circles(n_samples=num_points, factor=0.2, noise=0.1,
                        random_state=np.random.seed(10))
    x = x.reshape(-1, 2)
    return x, y.reshape(-1, 1)


def visualization():
    x, y = make_nonlinear_cla_data()
    pca = KernelPCA(n_components=2, kernel='rbf', gamma=2.)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title('Original', fontsize=15)
    plt.scatter(x[:, 0], x[:, 1], c=y)
    x = pca.fit_transform(x)
    plt.subplot(1, 3, 2)
    plt.title('Projection with two components', fontsize=15)
    plt.scatter(x[:, 0], x[:, 1], c=y)
    plt.subplot(1, 3, 3)
    plt.title('Projection with one component', fontsize=15)
    plt.scatter(x[:, 0], [0] * len(x), c=y)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    visualization()
