import numpy as np
import matplotlib.pyplot as plt


def normal_distribution(mu=0, sigma2=2., n=100):
    x = np.linspace(-5, 5, 2 * n)
    y = (1 / np.sqrt(2 * np.pi * sigma2)) * np.exp(-((x - mu) ** 2) / (2 * sigma2))
    return x, y


def visualization_normal_dis():
    x, y = normal_distribution(mu=0, sigma2=0.2)
    plt.plot(x, y, label=r'$\mu=0,\sigma^2=0.2$')
    x, y = normal_distribution(mu=0, sigma2=1)
    plt.plot(x, y, label=r'$\mu=0,\sigma^2=1$')
    x, y = normal_distribution(mu=0, sigma2=5)
    plt.plot(x, y, label=r'$\mu=0,\sigma^2=5$')
    x, y = normal_distribution(mu=-2, sigma2=0.5)
    plt.plot(x, y, label=r'$\mu=-2,\sigma^2=0.5$')

    plt.vlines(0, -0.10, 0.9, color='r',linestyles='--')
    plt.xlabel('x', fontsize=15)
    plt.ylabel(r'$\phi_{\mu,\sigma^2}(x)$', fontsize=15)
    plt.legend(fontsize=12)
    plt.tight_layout()  # 调整子图间距
    plt.show()


if __name__ == '__main__':
    visualization_normal_dis()
