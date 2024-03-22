import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles

#
if __name__ == '__main__':

    x, y = make_circles(n_samples=1000, noise=0.13, random_state=2, factor=0.1)

    marker = ['o', 's']
    plt.rcParams['ytick.direction'] = 'in'  # 刻度向内
    plt.rcParams['xtick.direction'] = 'in'  # 刻度向内
    for i in range(2):
        idx = np.where(y == i)[0]
        data = x[idx, :]
        X, Y = data[:, 0], data[:, 1]
        plt.scatter(X, Y, marker=marker[i])
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    marker = ['o', 's']
    for i in range(2):
        idx = np.where(y == i)[0]
        data = x[idx, :]
        X, Y = data[:, 0], data[:, 1]
        ax.scatter(X, Y, 1 / np.exp(2 * (np.abs(X) + np.abs(Y))), marker=marker[i], s=50)

    x = y = np.arange(-1, 1.5, 0.01)
    xx, yy = np.meshgrid(x, y)
    zz = 0. * (xx + yy) + 0.2
    ax.plot_surface(xx, yy, zz, color='black', alpha=0.7)

    plt.tight_layout()
    plt.show()
