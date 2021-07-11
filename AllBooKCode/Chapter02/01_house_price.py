import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def make_data():
    np.random.seed(20)
    x = np.random.rand(100) * 30 + 50  # square
    noise = np.random.rand(100) * 50
    y = x * 8 - 127  # price
    y = y - noise
    return x, y


def visualization(x, y):
    plt.xlabel('面积', fontsize=15)
    plt.ylabel('房价', fontsize=15)
    plt.scatter(x, y,c='black')
    plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
    plt.tight_layout()  # 调整子图间距
    plt.show()


if __name__ == '__main__':
    x, y = make_data()
    visualization(x, y)
