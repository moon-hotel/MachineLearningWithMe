"""
文件名: AllBooKCode/Chapter11/C04_PCA_vs_KPCA.py
创建时间: 2022/8/7
作 者: @空字符
公众号: @月来客栈
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import PCA
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt


def make_nonlinear_cla_data():
    num_points = 500
    x, y = make_circles(n_samples=num_points, factor=0.2, noise=0.1,
                        random_state=np.random.seed(10))
    x = x.reshape(-1, 2)
    return x, y.reshape(-1, 1)


def visualization():
    plt.figure(figsize=(15, 5), dpi=80)
    plt.subplot(1, 3, 1)
    plt.title('Original Projection', fontsize=15)
    x_orig, y = make_nonlinear_cla_data()
    plt.scatter(x_orig[:, 0], x_orig[:, 1], c=y)
    plt.tick_params(axis='x', labelsize=15)  # x轴刻度数字大小
    plt.tick_params(axis='y', labelsize=15)  # y轴刻度数字大小

    plt.subplot(1, 3, 2)
    plt.title('Projection via PCA', fontsize=15)
    pca = PCA(n_components=1)
    x = pca.fit_transform(x_orig)
    plt.scatter(x[:, 0], [0] * len(x), c=y)
    plt.tick_params(axis='x', labelsize=15)  # x轴刻度数字大小
    plt.tick_params(axis='y', labelsize=15)  # y轴刻度数字大小

    plt.subplot(1, 3, 3)
    plt.title('Projection via KernelPCA', fontsize=15)
    pca = KernelPCA(n_components=1, kernel='rbf', gamma=10)
    x = pca.fit_transform(x_orig)
    plt.scatter(x[:, 0], [0] * len(x), c=y)
    plt.tick_params(axis='x', labelsize=15)  # x轴刻度数字大小
    plt.tick_params(axis='y', labelsize=15)  # y轴刻度数字大小
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    visualization()
