from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def plot_surface():
    fig = plt.figure()
    ax = Axes3D(fig)
    X = np.arange(-4, 4, 0.25)
    Y = np.arange(-4, 4, 0.25)
    X, Y = np.meshgrid(X, Y)
    Z = X ** 2 + Y ** 2+2*Y + 5

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
    ax.set_zlabel(r'$J(w_1,w_2)$',fontsize=12)  # 坐标轴
    ax.set_ylabel(r'$w_2$',fontsize=12)
    ax.set_xlabel(r'$w_1$',fontsize=12)
    plt.show()


if __name__ == '__main__':
    plot_surface()
    X,Y = -1.6,2.2
    Z = X ** 2 + Y ** 2 + 2 * Y + 5
    print(Z)
