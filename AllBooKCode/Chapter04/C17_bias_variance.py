import numpy as np

import matplotlib.pyplot as plt
import mpl_toolkits.axisartist as axisartist

import numpy as np
import matplotlib.pyplot as plt


def bias_var():
    # Make some fake data.
    a = b = np.arange(0, 3, .02)
    c = np.exp(a)
    d = c[::-1]

    # Create plots with pre-defined labels.
    fig, ax = plt.subplots()
    ax.plot(a, c, 'k--')
    ax.plot(a, d, 'k:', )
    ax.plot(a, c + d, 'k')

    plt.annotate('方差', xy=(2.5, 9), fontsize=15)
    plt.annotate('偏差（平方）', xy=(2.2, 2.7), fontsize=15)
    plt.annotate('总误差', xy=(2, 15), fontsize=15)
    plt.vlines(1.5, 0, 20, linestyles='--')
    plt.annotate('最优模型', xy=(1.3, 10.2), fontsize=15, rotation=90)
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体 Windows
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.xticks([])
    plt.yticks([])

    plt.xlabel("模型复杂度", fontsize=15)
    plt.ylabel("误差", size=15)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    bias_var()
