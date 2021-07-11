import numpy as np
import matplotlib.pyplot as plt



def f(W1, W2):
    return (1 / 4) * W1 ** 2 + W2 ** 2


def f_grad(W1, W2, descent=True):
    grad = np.array([0.5 * W1, 2 * W2])
    if descent:
        grad *= -1
    learning_rate = 0.2
    return learning_rate * grad


def plot_countour():
    W1 = np.arange(-5, 5, 0.1)
    W2 = np.arange(-5, 5, 0.1)
    W1, W2 = np.meshgrid(W1, W2)
    J = f(W1, W2)
    fig, ax = plt.subplots(figsize=(6,6))
    CS = ax.contour(W1, W2, J, 9)
    ax.scatter(0, 0, c='black')
    p = np.array([-4.5, 4.5])  # 起始位置
    ax.scatter(p[0], p[1], c='black')
    for i in range(28):  # 梯度反方向，最速下降曲线
        q = f_grad(p[0], p[1])
        print("P{}:{}".format(i, p))
        ax.arrow(p[0], p[1], q[0], q[1], head_width=0.1, head_length=0.1, fc='black', ec='black')
        p += q  # 上一次的位置加上本次的梯度
    plt.annotate("反梯度方向,最速下降曲线", xy=(-4.7, -0.5), fontsize=14, c='black')

    p = np.array([0.3, 0.01])  # 起始位置
    for i in range(20):  # 梯度方向，最速上升曲线
        q = f_grad(p[0], p[1], descent=False)
        ax.arrow(p[0], p[1], q[0], q[1], head_width=0.1, head_length=0.05, fc='black', ec='black')
        p += q  # 上一次的位置加上本次的梯度
    plt.annotate("梯度方向,最速上升曲线", xy=(0.5, -0.5), fontsize=14, c='black')
    ax.clabel(CS, inline=2, fontsize=10)
    ax.set_xlabel(r'$w_1$', fontsize=15)
    ax.set_ylabel(r'$w_2$', fontsize=15)
    plt.rcParams['font.sans-serif'] = ['SimSun']  # 用来  正常显示中文标签
    plt.show()


if __name__ == '__main__':
    plot_countour()
