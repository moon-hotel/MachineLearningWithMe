import numpy as np
import matplotlib.pyplot as plt


def f_convex(W):
    return (1 / 4) * W ** 2


def f_convex_grad(W, ascent=False, learning_rate=0.2):
    grad = 0.5 * W
    if ascent:
        grad *= -1
    W = W - learning_rate * grad
    return grad, np.array([W, f_convex(W)])


def plot_countour():
    W = np.arange(-3, 3, 0.1)
    J = f_convex(W)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(W, J)
    plt.scatter(0, 0, c='r')
    p = np.array([-2.5, f_convex(-2.5)])  # 起始位置
    plt.scatter(p[0], p[1], c='r',label='learning rate = 0.4')

    for i in range(100):
        g, q = f_convex_grad(p[0], learning_rate=0.4)
        if np.abs(g - 0) < 0.001:
            break
        print("P{}:{},grad = {}".format(i, p, g))
        plt.arrow(p[0], p[1], q[0] - p[0], q[1] - p[1], head_width=0.07,
                  head_length=0.08, fc='r', ec='r', linewidth=1.5)
        p = q
    plt.ylabel('J', fontsize=15)
    plt.xlabel('w', fontsize=15)
    plt.legend(fontsize=12)

    plt.subplot(1, 2, 2)
    plt.plot(W, J)
    plt.scatter(0, 0, c='r')
    p = np.array([-2.5, f_convex(-2.5)])  # 起始位置
    plt.scatter(p[0], p[1], c='r',label='learning rate = 3.5')
    for i in range(100):  # 梯度反方向，最速下降曲线
        g, q = f_convex_grad(p[0], learning_rate=3.5)
        if np.abs(g - 0) < 0.001:
            break
        print("P{}:{},grad = {}".format(i, p, g))
        plt.arrow(p[0], p[1], q[0] - p[0], q[1] - p[1],
                  head_width=0.07, head_length=0.1, fc='r', ec='r', linewidth=1.5)
        p = q
    plt.ylabel('J', fontsize=15)
    plt.xlabel('w', fontsize=15)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    plot_countour()
