import numpy as np
import matplotlib.pyplot as plt


def f(x):
    # 圆： (x-2)^2 + y^2 = 1
    return np.sqrt(1 - (x - 2) ** 2)


def f_prime(x):
    return (2 - x) / np.sqrt(1 - (x - 2) ** 2)


def kx_plus_b(x, y, k, offset_left=1, offset_right=1):
    b = y - k * x
    x_left = x - offset_left
    y_left = k * x_left + b
    x_right = x + offset_right
    y_right = k * x_right + b
    return [[x_left, x_right], [y_left, y_right]]


def visualization():
    x = np.linspace(1, 3, 200)
    y = f(x)
    plt.figure(figsize=(7, 3.5))
    plt.rcParams['ytick.direction'] = 'in'  # 刻度向内
    plt.rcParams['xtick.direction'] = 'in'  # 刻度向内
    plt.tick_params(axis='x', labelsize=15)  # x轴刻度数字大小
    plt.tick_params(axis='y', labelsize=15)  # y轴刻度数字大小
    plt.plot(x, y, label=r'$(x-2)^2+y^2=1$', c='black')
    plt.scatter(2.5, f(2.5), c='black')
    line1 = kx_plus_b(2.5, f(2.5), f_prime(2.5), 0.5, 0.5)
    line2 = kx_plus_b(2.5, f(2.5), -1 / f_prime(2.5), 0.2, 0)
    p1 = [line2[0][0], line2[1][0]]
    p2 = [line2[0][1], line2[1][1]]
    plt.arrow(p2[0], p2[1], -(p2[0] - p1[0]), -(p2[1] - p1[1]),
              head_width=0.03, head_length=0.05, fc='black', ec='black')
    plt.annotate(r'$\overrightarrow{m}$', xy=(2.35, 0.4), fontsize=15, c='black')
    plt.annotate(r'$\overrightarrow{n}$', xy=(2.7, 0.8), fontsize=15, c='black')
    plt.annotate(r'$P$', xy=(2.5, 0.92), fontsize=15, c='black')
    plt.plot(line1[0], line1[1], c='black')
    plt.legend(fontsize=15)
    plt.show()


if __name__ == '__main__':
    visualization()
