import numpy as np
import matplotlib.pyplot as plt


def f_non_convex(W):
    return 0.2 * W ** 2 + 2 * (np.sin(2 * W))


def f_non_convex_grad(W, ascent=False, learning_rate=0.2):
    grad = 0.4 * W + 2 * np.cos(2 * W) * 2
    if ascent:
        grad *= -1
    W = W - learning_rate * grad
    return grad, np.array([W, f_non_convex(W)])


def plot_countour():
    W = np.linspace(-5, 4, 800)
    J = f_non_convex(W)
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(W, J)
    plt.scatter(-0.7584, -1.8747, marker='*', c='red', s=80, label=r'$\alpha= 0.02,ite = 20$')  # 非实际计算
    p = np.array([-4.8, f_non_convex(-4.8)])  # 起始位置
    plt.scatter(p[0], p[1])
    plt.legend(fontsize=15, loc='upper center')

    for i in range(20):
        g, q = f_non_convex_grad(p[0], learning_rate=0.02)
        print("P{}:{},grad = {}".format(i, p, g))
        plt.arrow(p[0], p[1], q[0] - p[0], q[1] - p[1], head_width=0.1,
                  head_length=0.1, linewidth=2)
        p = q
        plt.scatter(p[0], p[1])
    plt.ylabel('J(w)', fontsize=15)
    plt.xlabel('w', fontsize=15)

    plt.subplot(1, 3, 2)
    plt.plot(W, J)
    plt.scatter(-0.7584, -1.8747, marker='*', c='red', s=80, label=r'$\alpha= 0.4,ite = 4$')  # 非实际计算
    p = np.array([-4.8, f_non_convex(-4.8)])  # 起始位置
    plt.scatter(p[0], p[1])
    for i in range(4):
        g, q = f_non_convex_grad(p[0], learning_rate=0.4)
        print("P{}:{},grad = {}".format(i, p, g))

        plt.arrow(p[0], p[1], q[0] - p[0], q[1] - p[1], head_width=0.1,
                  head_length=0.1, linewidth=2)
        p = q
        plt.scatter(p[0], p[1])
    plt.ylabel('J(w)', fontsize=15)
    plt.xlabel('w', fontsize=15)
    plt.legend(fontsize=15, loc='upper center')

    plt.subplot(1, 3, 3)
    plt.plot(W, J)
    plt.scatter(-0.7584, -1.8747, marker='*', c='red', s=80, label=r'$\alpha= 0.6,ite = 20$')  # 非实际计算
    p = np.array([-4.8, f_non_convex(-4.8)])  # 起始位置
    plt.scatter(p[0], p[1])
    for i in range(20):
        g, q = f_non_convex_grad(p[0], learning_rate=0.6)
        print("P{}:{},grad = {}".format(i, p, g))
        plt.arrow(p[0], p[1], q[0] - p[0], q[1] - p[1], head_width=0.1,
                  head_length=0.1, linewidth=2)
        p = q
        plt.scatter(p[0], p[1])
    plt.ylabel('J(w)', fontsize=15)
    plt.xlabel('w', fontsize=15)
    plt.legend(fontsize=15, loc='upper center')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_countour()
