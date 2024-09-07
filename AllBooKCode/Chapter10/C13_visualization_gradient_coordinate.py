import numpy as np
import matplotlib.pyplot as plt


def f_grad2(W1, W2, descent=True):
    grad = -np.array([2. * W1 - W2 - 1, 9 * W2 - W1 + 4.])
    if descent:
        grad *= -1
    learning_rate = 0.05
    return learning_rate * grad


def cost(W1, W2):
    J = (1 / 2) * (W1 - 1) ** 2 + (2 * W2 + 1) ** 2 + (1 / 2) * (W1 - W2) ** 2
    return -J


def plot_countour():
    W1 = np.arange(-4, 4, 0.25)
    W2 = np.arange(-4, 4, 0.25)
    W1, W2 = np.meshgrid(W1, W2)
    plt.figure(figsize=(8, 8), dpi=80)
    plt.rcParams['ytick.direction'] = 'in'  # 刻度向内
    plt.rcParams['xtick.direction'] = 'in'  # 刻度向内
    J = cost(W1, W2)
    CS = plt.contour(W1, W2, J, 16)
    plt.clabel(CS, inline=2, fontsize=10)
    plt.annotate(r'$(0.26,-0.41)$', xy=(-0.6, -0.9), fontsize=15)
    plt.xlabel(r'$w_1$', fontsize=18)
    plt.ylabel(r'$w_2$', fontsize=18)

    # --------------  坐标下降算法
    p = np.array([-3.8, 2.])  # 起始位置
    q = np.array([0., 0.])  # 初始化
    plt.scatter(p[0], p[1])
    for i in range(50):
        q[0] = 0.5 * (p[1] + 1)
        plt.arrow(p[0], p[1], q[0] - p[0], 0., head_width=0.1, head_length=0.1, )
        p[0] = q[0]

        q[1] = (1 / 9.) * (p[0] - 4.)
        plt.arrow(p[0], p[1], 0., q[1] - p[1], head_width=0.1, head_length=0.1, )
        p[1] = q[1]

        # print("P{}:{}, J={}".format(i, p, cost(p[0], p[1])))
    #
    # # ------------  梯度下降算法
    p = np.array([-3.8, 2.])  # 起始位置
    plt.scatter(p[0], p[1])
    for i in range(50):  #
        q = f_grad2(p[0], p[1], descent=False)
        plt.arrow(p[0], p[1], q[0], q[1], head_width=0.1, head_length=0.1, )
        p += q  # 上一次的位置加上本次的梯度
        print("P{}:{}, J={}".format(i, p, cost(p[0], p[1])))
    plt.show()


if __name__ == '__main__':
    plot_countour()
    print(cost(5 / 17, -7 / 17))
