import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_data():
    data = pd.read_csv('./data/admitted_or_not.txt', names=['exam1', 'exam2', 'label'])
    data = np.array(data)
    X = data[:, :2]  # 取前两列
    y = np.array(data[:, -1:])  # 取最后一列
    return X, y


def visualization(X, y):
    positive = np.where(y == 1)[0]
    negative = np.where(y == 0)[0]
    plt.scatter(X[positive, 0], X[positive, 1], s=30, c='b', marker='o', label='Admitted')
    plt.scatter(X[negative, 0], X[negative, 1], s=30, c='r', marker='o', label='Not Admitted')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    x, y = load_data()
    visualization(x, y)
    print(x.shape)

    # 你需要完成的工作：
    # 1. 利用逻辑回归算法（自己实现的或者sklearn)完成对数据集的分类并输出准确率；
    # 2. 可视化目标函数的损失值
    # 3. 仿照之前示例，绘制出决策曲面（可选）
