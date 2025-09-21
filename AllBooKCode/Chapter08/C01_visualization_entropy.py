import numpy as np
import matplotlib.pyplot as plt


def visualization():
    p = np.linspace(0, 1, 100)
    q = 1 - p
    h = -1. * (p * np.log2(p + 0.0001) + q * np.log2(q + 0.0001))
    # +0.0001  是平滑处理，防止 p 或 q 为 0 的情况。
    plt.rcParams['ytick.direction'] = 'in'  # 刻度向内
    plt.rcParams['xtick.direction'] = 'in'  # 刻度向内
    plt.plot(p, h, c='black')
    plt.show()


if __name__ == '__main__':
    visualization()
