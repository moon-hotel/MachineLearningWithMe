import matplotlib.pyplot as plt
import numpy as np


def visualization():
    a = plt.gca()
    plt.rcParams['ytick.direction'] = 'in'  # 刻度向内
    plt.rcParams['xtick.direction'] = 'in'  # 刻度向内
    a.spines['right'].set_visible(False)
    a.spines['top'].set_visible(False)
    a.spines['left'].set_position(('data', 0))
    a.spines['bottom'].set_position(('data', 0))
    points = np.array([[2, 3], [1, 1]])
    plt.scatter(points[:, 0], points[:, 1])
    plt.plot([0, 3], [3, 0])
    plt.annotate(r'$A(2,3)$', xy=(2.1, 3), fontsize=12)
    plt.annotate(r'$B(1,1)$', xy=(1.1, 1), fontsize=12)
    plt.show()


if __name__ == '__main__':
    visualization()
