import numpy as np
import matplotlib.pyplot as plt


def f_grad1(W1, W2, descent=True):
    grad = np.array([2 * W1, 2 * W2])
    if descent:
        grad *= -1
    learning_rate = 0.2
    return learning_rate * grad


def f_grad2(W1, W2, descent=True):
    grad = np.array([1 / 3 * W1, 2 * W2])
    if descent:
        grad *= -1
    learning_rate = 0.2
    return learning_rate * grad


def plot_countour():
    W1 = np.arange(-4, 4, 0.25)
    W2 = np.arange(-4, 4, 0.25)
    W1, W2 = np.meshgrid(W1, W2)
    plt.figure(figsize=(10, 5), dpi=80)
    plt.subplot(1, 2, 1)
    J = W1 ** 2 + W2 ** 2
    CS = plt.contour(W1, W2, J, 10)
    plt.clabel(CS, inline=2, fontsize=12)
    plt.scatter(0, 0, s=80, c='black')
    plt.xlabel(r'$w_1$', fontsize=15)
    plt.ylabel(r'$w_2$', fontsize=15)

    p = np.array([-3.5, 3.5])  # 起始位置
    plt.scatter(p[0], p[1], c='black')
    for i in range(8):  # 梯度反方向，最速下降曲线
        q = f_grad1(p[0], p[1])
        print("P{}:{}".format(i, p))
        plt.arrow(p[0], p[1], q[0], q[1], head_width=0.1, head_length=0.1, fc='r', ec='r')
        p += q  # 上一次的位置加上本次的梯度

    plt.subplot(1, 2, 2)
    J = (1 / 6) * W1 ** 2 + W2 ** 2
    CS = plt.contour(W1, W2, J, 16)
    plt.clabel(CS, inline=2, fontsize=10)
    plt.scatter(0, 0, s=80, c='black')
    plt.xlabel(r'$w_1$', fontsize=15)
    plt.ylabel(r'$w_2$', fontsize=15)

    p = np.array([-3.5, 3.5])  # 起始位置
    plt.scatter(p[0], p[1],c='black')
    for i in range(40):  # 梯度反方向，最速下降曲线
        q = f_grad2(p[0], p[1])
        print("P{}:{}".format(i, p))
        plt.arrow(p[0], p[1], q[0], q[1], head_width=0.1, head_length=0.1, fc='r', ec='r')
        p += q  # 上一次的位置加上本次的梯度

    plt.tight_layout()  # 调整子图间距
    plt.show()


if __name__ == '__main__':
    plot_countour()
