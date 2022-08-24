import matplotlib.pyplot as plt
import numpy as np


def visualization():
    a = plt.gca()
    a.spines['right'].set_visible(False)
    a.spines['top'].set_visible(False)
    a.spines['left'].set_position(('data', 0))
    a.spines['bottom'].set_position(('data', 0))
    points_neg = np.array([[2, 3], [1, 2]])
    points_pos = np.array([[5,4],[6,3]])
    plt.scatter(points_neg[:, 0], points_neg[:, 1],c='b')
    plt.scatter(points_pos[:, 0], points_pos[:, 1],c='r')
    plt.plot([-1, 8], [8, -1],c='black')
    plt.annotate(r'$A(2,3)$', xy=(2.1, 3), fontsize=12, c='b')
    plt.annotate(r'$B(1,2)$', xy=(1.1, 1.7), fontsize=12, c='b')
    plt.annotate(r'$C(6,3)$', xy=(6.1, 2.8), fontsize=12, c='r')
    plt.annotate(r'$D(5,4)$', xy=(5.1, 4), fontsize=12, c='r')
    plt.show()


if __name__ == '__main__':
    visualization()
