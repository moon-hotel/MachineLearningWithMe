import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return np.log((1 - x) / (x + 1e-5))


if __name__ == '__main__':
    x = np.linspace(0, 1, 200)
    plt.plot(x, f(x))
    a = plt.gca()
    a.spines['right'].set_visible(False)
    a.spines['top'].set_visible(False)
    a.spines['left'].set_position('center')
    plt.xlabel(r"$err^{(m)}$", fontsize=14)
    plt.ylabel(r"$\alpha^{(m)}$", fontsize=14)
    plt.show()
