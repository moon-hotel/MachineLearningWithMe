import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

"""
对特征标准之前和之后的数据进行可视化对比
"""


def standarlization(X):
    # Z-score 标准化，通过减去均值并除以标准差来将数据转换为标准正态分布，使得数据的均值为0，标准差为1。
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return (X - mean) / std


def standarlization_minmax(X):
    # min-max 标准化，将数据线性缩放到一个特定的范围内，通常是[0, 1]或[-1, 1]
    min = X.min(axis=0)
    max = X.max(axis=0)
    return (X - min) / (max - min)


def make_data():
    np.random.seed(2020)
    # x1 = np.random.randint(3, 8, (50, 1))
    # x2 = np.random.randint(2, 8, (50, 1))
    # x = np.hstack((x1, x2))
    x = np.random.randn(50, 2)+2.5
    x_standarlized = standarlization(x)
    x_standarlized_minmax = standarlization_minmax(x)
    return x, x_standarlized, x_standarlized_minmax


def visualization():
    x, x_standarlized, x_standarlized_minmax = make_data()
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(x[:, 0], x[:, 1], label='原始数据')
    plt.scatter(x_standarlized[:, 0], x_standarlized[:, 1], label='标准化后数据')

    # plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.tick_params(axis='x', labelsize=15)  # x轴刻度数字大小
    plt.tick_params(axis='y', labelsize=15)  # y轴刻度数字大小
    plt.title('去均值标准化',fontsize=15)
    plt.legend(fontsize=15)

    plt.subplot(1, 2, 2)
    plt.scatter(x[:, 0], x[:, 1], label='原始数据')
    plt.scatter(x_standarlized_minmax[:, 0], x_standarlized_minmax[:, 1], label='标准化后数据')
    plt.title('最小-最大值标准化',fontsize=15)
    plt.tick_params(axis='x', labelsize=15)  # x轴刻度数字大小
    plt.tick_params(axis='y', labelsize=15)  # y轴刻度数字大小
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体

    # plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.legend(fontsize=15)
    plt.show()


if __name__ == '__main__':
    visualization()
    # X, y = make_circles(n_samples=100, noise=0.05, random_state=2024, factor=0.5)
    # X = X  * 5
    # print(X.shape, y.shape)
    # plt.scatter(X[:, 0], X[:, 1], c=y)
    # plt.show()
