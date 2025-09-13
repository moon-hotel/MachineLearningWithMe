import numpy as np
import matplotlib.pyplot as plt


def make_data():
    np.random.seed(20)
    x = np.random.rand(100) * 30 + 50  # square
    noise = np.random.rand(100) * 50
    y = x * 8 - 127  # price
    y = y - noise
    return x, y


def visualization(x, y):
    plt.rcParams['ytick.direction'] = 'in'  # 刻度向内
    plt.rcParams['xtick.direction'] = 'in'  # 刻度向内
    plt.xlabel('面积', fontsize=16)
    plt.ylabel('房价', fontsize=16)
    plt.tick_params(axis='x', labelsize=15)  # x轴刻度数字大小
    plt.tick_params(axis='y', labelsize=15)  # y轴刻度数字大小
    plt.scatter(x, y, c='black')
    plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
    # plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.tight_layout()  # 调整子图间距
    plt.show()


if __name__ == '__main__':
    x, y = make_data()
    visualization(x, y)
