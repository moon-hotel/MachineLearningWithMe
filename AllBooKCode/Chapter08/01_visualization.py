import numpy as np
import matplotlib.pyplot as plt


def visualization():
    p = np.linspace(0, 1, 100)
    q = 1 - p
    h = -1. * (p * np.log2(p) + q * np.log2(q))
    plt.rcParams['ytick.direction'] = 'in'  # 刻度向内
    plt.rcParams['xtick.direction'] = 'in'  # 刻度向内
    plt.plot(p, h, c='black')
    plt.show()


if __name__ == '__main__':
    visualization()
