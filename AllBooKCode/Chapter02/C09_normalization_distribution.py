import numpy as np
import matplotlib.pyplot as plt


def normal_distribution(mu=0, sigma2=2., n=100):
    x = np.linspace(-5, 5, 2 * n)
    y = (1 / np.sqrt(2 * np.pi) * sigma2) * np.exp(-((x - mu) ** 2) / (2 * sigma2 ** 2))
    return x, y


def visualization_normal_dis():
    x, y = normal_distribution(mu=0, sigma2=1)
    plt.rcParams['xtick.direction'] = 'in'  # 刻度向内
    plt.rcParams['ytick.direction'] = 'in'  # 刻度向内
    plt.plot(x, y, label=r'$f(x)$', c='black')
    plt.vlines(0, 0, 0.42, color='black', linestyles='--')
    plt.xlabel("误差", fontsize=15)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False
    # plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.tick_params(axis='x', labelsize=15)  # x轴刻度数字大小
    plt.tick_params(axis='y', labelsize=15)  # y轴刻度数字大小

    plt.legend(fontsize=16)
    plt.tight_layout()  # 调整子图间距
    plt.show()


if __name__ == '__main__':
    visualization_normal_dis()
