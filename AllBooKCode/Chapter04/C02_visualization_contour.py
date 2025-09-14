import numpy as np
import matplotlib.pyplot as plt


def plot_countour():
    W1 = np.arange(-4, 4, 0.25)
    W2 = np.arange(-4, 4, 0.25)
    W1, W2 = np.meshgrid(W1, W2)
    J = W1 ** 2 + W2 ** 2 + 5
    plt.rcParams['ytick.direction'] = 'in'  # 刻度向内
    plt.rcParams['xtick.direction'] = 'in'  # 刻度向内
    fig, ax = plt.subplots()
    CS = ax.contour(W1, W2, J, 10, colors='black')
    ax.clabel(CS, inline=2, fontsize=10)
    # ax.set_title('Simplest default with labels')
    # ax.set_xlim(-5, 5)
    # ax.set_ylim(-5, 5)
    ax.set_xlabel(r'$w_1$', fontsize=16)
    ax.set_ylabel(r'$w_2$', fontsize=16)
    plt.tick_params(axis='x', labelsize=15)  # x轴刻度数字大小
    plt.tick_params(axis='y', labelsize=15)  # y轴刻度数字大小
    plt.show()


if __name__ == '__main__':
    plot_countour()
