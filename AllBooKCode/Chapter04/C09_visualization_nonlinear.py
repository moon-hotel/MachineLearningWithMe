import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles

np.random.seed(10)


def make_nonlinear_reg_data():
    np.random.seed(10)
    num_points = 100
    x = np.linspace(-5, 5, num_points)
    y = x ** 2 + np.random.randn(num_points)
    return x, y


def make_nonlinear_cla_data():
    num_points = 200
    x, y = make_circles(num_points, factor=0.5, noise=0.06, random_state=np.random.seed(10))
    return x, y


def visualization():
    plt.figure(figsize=(12, 5))
    plt.rcParams['ytick.direction'] = 'in'  # 刻度向内
    plt.rcParams['xtick.direction'] = 'in'  # 刻度向内
    plt.subplot(1, 2, 1)
    x, y = make_nonlinear_reg_data()
    plt.scatter(x, y, s=50)
    plt.tick_params(axis='x', labelsize=20)  # x轴刻度数字大小
    plt.tick_params(axis='y', labelsize=20)  # y轴刻度数字大小
    plt.subplot(1, 2, 2)
    x, y = make_nonlinear_cla_data()
    neg, pos = [], []
    for i in range(len(x)):
        if y[i] == 0:
            neg.append(x[i, :])
        else:
            pos.append(x[i, :])
    neg, pos = np.vstack(neg), np.vstack(pos)
    plt.scatter(neg[:, 0], neg[:, 1], marker='s', s=50)
    plt.scatter(pos[:, 0], pos[:, 1], marker='o', s=50)
    plt.tight_layout()
    plt.tick_params(axis='x', labelsize=20)  # x轴刻度数字大小
    plt.tick_params(axis='y', labelsize=20)  # y轴刻度数字大小
    plt.show()


if __name__ == '__main__':
    visualization()
