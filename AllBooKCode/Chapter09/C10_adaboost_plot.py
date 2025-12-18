import numpy as np
import matplotlib.pyplot as plt


def f(x, K=2):
    return np.log((1 - x) / (x + 1e-5)) + np.log(K - 1)


if __name__ == '__main__':
    x = np.linspace(0, 1, 200)
    plt.plot(x, f(x), label='K=2')
    plt.plot(x, f(x, 3), label='K=3')
    plt.plot(x, f(x, 4), label='K=4')
    a = plt.gca()
    a.spines['right'].set_visible(False)
    a.spines['top'].set_visible(False)
    a.spines['left'].set_position('center')
    plt.xlabel(r"$err^{(m)}$", fontsize=14)
    plt.ylabel(r"$\alpha^{(m)}$", fontsize=14)
    plt.legend(loc='best')
    plt.show()
