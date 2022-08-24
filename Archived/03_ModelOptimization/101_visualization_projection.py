import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


def plot_contour():
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    W1 = np.arange(-4, 4, 0.25)
    W2 = np.arange(-4, 4, 0.25)
    W1, W2 = np.meshgrid(W1, W2)
    J = W1 ** 2 + W2 ** 2 + 5


    # Plot the 3D surface
    ax.plot_surface(W1, W2, J, rstride=1, cstride=1, cmap='rainbow')

    cset = ax.contourf(W1, W2, J, zdir='z', offset=-5, cmap=cm.coolwarm)
    # cset = ax.contourf(X, Y, Z, zdir='x', offset=-40, cmap=cm.coolwarm)
    # cset = ax.contourf(X, Y, Z, zdir='y', offset=40, cmap=cm.coolwarm)

    # ax.set_xlim(-5, 5)
    # ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 35)

    ax.set_xlabel(r'$w_1$', fontsize=15)
    ax.set_ylabel(r'$w_2$', fontsize=15)
    ax.set_zlabel(r'$J(w_1,w_2)$', fontsize=15)

    plt.show()


if __name__ == '__main__':
    plot_contour()
