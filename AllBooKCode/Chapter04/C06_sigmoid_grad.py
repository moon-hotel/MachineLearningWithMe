import numpy as np
import matplotlib.pyplot as plt


def g(z):
    return 1 / (1 + np.exp(-z))


def visualization():
    x = np.linspace(-10, 10, 100)
    y = g(x)
    y_prime = g(x) * (1 - g(x))
    a = plt.gca()
    a.spines['right'].set_visible(False)
    a.spines['top'].set_visible(False)
    a.spines['left'].set_position('center')

    plt.plot(x, y, label=r'$g(z)$', c='black', linestyle='-')
    plt.plot(x, y_prime, label=r'$g(z)^{\prime}$', c='black', linestyle='--')
    plt.hlines(1, -10, 10, linestyles='--', colors='black')
    plt.hlines(0, -10, 10, linestyles='--', colors='black')
    plt.legend(fontsize=13, loc='center left')
    plt.rcParams['ytick.direction'] = 'in'  # 刻度向内
    plt.rcParams['xtick.direction'] = 'in'  # 刻度向内
    plt.tick_params(axis='x', labelsize=15)  # x轴刻度数字大小
    plt.tick_params(axis='y', labelsize=15)  # y轴刻度数字大小
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    # plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.show()


if __name__ == '__main__':
    visualization()
