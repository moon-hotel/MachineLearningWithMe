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
    plt.figure(figsize=(15, 5))
    plt.rcParams['ytick.direction'] = 'in'  # 刻度向内
    plt.rcParams['xtick.direction'] = 'in'  # 刻度向内
    plt.subplot(1, 3, 1)
    plt.plot(W, J, c='black')
    plt.scatter(0, 0, marker='*', c='black', s=80, )
    p = np.array([-2.5, f_convex(-2.5)])  # 起始位置
    plt.scatter(p[0], p[1], label=r'$\alpha= 0.4,ite = 12$', c='black')

    for i in range(12):
        g, q = f_convex_grad(p[0], learning_rate=0.4)
        if np.abs(g - 0) < 0.001:
            break
        print("P{}:{},grad = {}".format(i, p, g))
        plt.arrow(p[0], p[1], q[0] - p[0], q[1] - p[1], head_width=0.07,
                  head_length=0.08, linewidth=1.5, color='black')
        p = q
    plt.ylabel('J(w)', fontsize=15)
    plt.xlabel('w', fontsize=15)
    plt.legend(fontsize=15, loc='upper center')

    # -----------------------------------------------
    plt.subplot(1, 3, 2)
    plt.rcParams['ytick.direction'] = 'in'  # 刻度向内
    plt.rcParams['xtick.direction'] = 'in'  # 刻度向内
    plt.plot(W, J, c='black')
    plt.scatter(0, 0, marker='*', c='black', s=80)
    p = np.array([-2.5, f_convex(-2.5)])  # 起始位置
    plt.scatter(p[0], p[1], label=r'$\alpha= 3.5,ite = 12$', c='black')
    for i in range(12):  # 梯度反方向，最速下降曲线
        g, q = f_convex_grad(p[0], learning_rate=3.5)
        if np.abs(g - 0) < 0.001:
            break
        print("P{}:{},grad = {}".format(i, p, g))
        plt.arrow(p[0], p[1], q[0] - p[0], q[1] - p[1],
                  head_width=0.07, head_length=0.1, linewidth=1.5, color='black')
        p = q
    plt.ylabel('J(w)', fontsize=15)
    plt.xlabel('w', fontsize=15)
    plt.legend(fontsize=15, loc='upper center')

    # -----------------------------------------------
    plt.subplot(1, 3, 3)
    plt.rcParams['ytick.direction'] = 'in'  # 刻度向内
    plt.rcParams['xtick.direction'] = 'in'  # 刻度向内
    plt.plot(W, J, c='black')
    plt.scatter(0, 0, marker='*', c='black', s=80)
    p = np.array([-2.5, f_convex(-2.5)])  # 起始位置
    plt.scatter(p[0], p[1], label=r'$\alpha= 4.1,ite = 12$', c='black')
    for i in range(12):  # 梯度反方向，最速下降曲线
        g, q = f_convex_grad(p[0], learning_rate=4.1)
        if np.abs(g - 0) < 0.001:
            break
        print("P{}:{},grad = {}".format(i, p, g))
        plt.arrow(p[0], p[1], q[0] - p[0], q[1] - p[1],
                  head_width=0.07, head_length=0.1, linewidth=1.5, color='black')
        p = q
    plt.ylabel('J(w)', fontsize=15)
    plt.xlabel('w', fontsize=15)
    plt.ylim(-0.1, 2.3)
    plt.xlim(-3, 3)
    plt.legend(fontsize=15, loc='upper center')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_countour()
