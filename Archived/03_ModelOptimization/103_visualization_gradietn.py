import numpy as np
import matplotlib.pyplot as plt


def plot_countour():
    W1 = np.arange(-4, 4, 0.25)
    W2 = np.arange(-4, 4, 0.25)
    W1, W2 = np.meshgrid(W1, W2)
    plt.figure(figsize=(10, 5), dpi=80)
    plt.subplot(1,2,1)
    J = W1 ** 2 + W2 ** 2
    CS = plt.contour(W1, W2, J, 10)
    plt.clabel(CS, inline=2, fontsize=12)
    plt.scatter(0,0,c='r')
    plt.xlabel(r'$w_1$',fontsize=15)
    plt.ylabel(r'$w_2$',fontsize=15)

    plt.subplot(1,2,2)
    J = (1/6) * W1 ** 2 +  W2 ** 2
    CS = plt.contour(W1, W2, J, 16)
    plt.clabel(CS, inline=2, fontsize=10)
    plt.scatter(0,0,c='r')
    plt.xlabel(r'$w_1$',fontsize=15)
    plt.ylabel(r'$w_2$',fontsize=15)

    plt.tight_layout()  # 调整子图间距
    plt.show()


if __name__ == '__main__':
    plot_countour()
