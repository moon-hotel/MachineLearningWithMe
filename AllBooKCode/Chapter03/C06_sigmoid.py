import numpy as np
import matplotlib.pyplot as plt


def g(z):
    return 1 / (1 + np.exp(-z))


def visualization():
    x = np.linspace(-10, 10, 100)
    y = g(x)
    plt.rcParams['ytick.direction'] = 'in'  # 刻度向内
    plt.rcParams['xtick.direction'] = 'in'  # 刻度向内
    # plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False
    a = plt.gca()
    a.spines['right'].set_visible(False)
    a.spines['top'].set_visible(False)
    a.spines['left'].set_position('center')

    plt.plot(x, y, label=r'$g(z)=\frac{1}{1+e^{-z}}$', c='black')
    plt.hlines(1, -10, 10, linestyles='--', colors='black')
    plt.hlines(0, -10, 10, linestyles='--', colors='black')
    plt.legend(fontsize=15, loc='center left')
    plt.tick_params(axis='x', labelsize=15)  # x轴刻度数字大小
    plt.tick_params(axis='y', labelsize=15)  # y轴刻度数字大小
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    visualization()
