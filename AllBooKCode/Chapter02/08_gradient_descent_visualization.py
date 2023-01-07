from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def cost_function(w1, w2):
    J = w1 ** 2 + w2 ** 2 + 2 * w2 + 5
    return J


def compute_gradient(w1, w2):
    return [2 * w1, 2 * w2 + 2]


def gradient_descent():
    w1, w2 = -2, 3
    jump_points = [[w1, w2]]
    costs = [cost_function(w1, w2)]
    step = 0.1
    print("P:({},{})".format(w1, w2), end=' ')
    for i in range(20):
        gradients = compute_gradient(w1, w2)
        w1 = w1 - step * gradients[0]
        w2 = w2 - step * gradients[1]
        jump_points.append([w1, w2])
        costs.append(cost_function(w1, w2))
        print("P{}:({},{})".format(i + 1, round(w1, 3), round(w2, 3)), end=' ')
    return jump_points, costs


def plot_surface_and_jump_points(jump_points, costs):
    fig = plt.figure()
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    w1 = np.arange(-4, 4, 0.25)
    w2 = np.arange(-4, 4, 0.25)
    w1, w2 = np.meshgrid(w1, w2)
    J = w1 ** 2 + w2 ** 2 + 2 * w2 + 5

    ax.plot_surface(w1, w2, J, rstride=1, cstride=1,
                    alpha=0.3, cmap='rainbow')
    ax.set_zlabel(r'$J(w_1,w_2)$', fontsize=12)  # 坐标轴
    ax.set_ylabel(r'$w_2$', fontsize=12)
    ax.set_xlabel(r'$w_1$', fontsize=12)

    jump_points = np.array(jump_points)
    ax.scatter3D(jump_points[:, 0], jump_points[:, 1], costs, c='black', s=50)
    plt.show()


if __name__ == '__main__':
    jump_points, costs = gradient_descent()
    plot_surface_and_jump_points(jump_points, costs)
