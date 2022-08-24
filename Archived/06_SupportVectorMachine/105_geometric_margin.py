import matplotlib.pyplot as plt
import numpy as np


def visualization():
    a = plt.gca()
    a.spines['right'].set_visible(False)
    a.spines['top'].set_visible(False)
    a.spines['left'].set_position(('data', 0))
    a.spines['bottom'].set_position(('data', 0))
    points = np.array([[1.3, 1.3], [1, 1]])

    plt.plot([0, 2], [2, 0], c='black')  # x+y = 2
    plt.plot([1, 1.3], [1, 1.3], c='black')  # x-y = 0
    plt.scatter(points[:, 0], points[:, 1],c='black')
    plt.arrow(0.7, 1.3, 1.2 - 0.7,  1.8- 1.3, head_width=0.05,  # y = x + 1
              head_length=0.05, fc='r', ec='r', linewidth=1.5)
    plt.xticks([])
    plt.yticks([])
    plt.annotate(r'$A$', xy=(1.4, 1.3), fontsize=12, c='r')
    plt.annotate(r'$B$', xy=(0.9, 0.9), fontsize=12, c='r')
    plt.annotate(r'$\overrightarrow{W}$', xy=(0.9, 1.7), fontsize=14, c='r')
    plt.annotate(r'$\gamma^{(i)}$', xy=(1.2, 1), fontsize=14, c='black')
    plt.show()


if __name__ == '__main__':
    visualization()
