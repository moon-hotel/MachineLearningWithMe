import numpy as np
import matplotlib.pyplot as plt


def plot_countour():
    W1 = np.arange(-3, 4, 0.25)
    W2 = np.arange(-3, 4, 0.25)
    W1, W2 = np.meshgrid(W1, W2)
    plt.figure(figsize=(8, 8), dpi=80)

    J = (1 / 6) * (W1 - 0.5) ** 2 + (W2 - 0.5) ** 2
    CS = plt.contour(W1, W2, J, 7, linestyles='--')
    plt.clabel(CS, inline=2, fontsize=10)
    plt.scatter(1 / 2, 1 / 2, )
    plt.annotate(r'$(0.5,0.5)$', xy=(0.6, 0.6), fontsize=18)

    J = (1 / 6) * (W1 - 0.5) ** 2 + (W2 - 0.5) ** 2 + (W1 ** 2 + W2 ** 2)
    CS = plt.contour(W1, W2, J, 7, )
    plt.clabel(CS, inline=2, fontsize=10)
    plt.scatter(1 / 16, 1 / 4, )
    plt.annotate(r'$(0.0625,0.25)$', xy=(-1.8, 0.2), fontsize=18)

    plt.scatter(0, 0, )
    plt.annotate(r'$(0,0)$', xy=(0.1, -0.3), fontsize=16)
    plt.xlabel(r'$w_1$', fontsize=15)
    plt.ylabel(r'$w_2$', fontsize=15)

    plt.tight_layout()  # 调整子图间距
    plt.show()


if __name__ == '__main__':
    plot_countour()
