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
    plt.plot(x, y_prime, label=r'$g(z)^{\prime}$', linestyle='--')
    plt.hlines(1, -10, 10, linestyles='--')
    plt.hlines(0, -10, 10, linestyles='--')
    plt.legend(fontsize=13, loc='center left')
    plt.show()


if __name__ == '__main__':
    visualization()
