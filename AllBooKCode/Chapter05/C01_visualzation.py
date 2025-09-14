"""
文件名: AllBooKCode/Chapter05/C01_visualzation.py
作 者: @空字符
B 站: @月来客栈Moon https://space.bilibili.com/392219165
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
油 管: @月来客栈
小红书: @月来客栈
公众号: @月来客栈
代码仓库: https://github.com/moon-hotel/MachineLearningWithMe
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.font_manager as fm


def visualization():
    num_points = 300
    plt.figure(figsize=(6, 6), dpi=120)
    centers = [[1, 1], [1.8, 1.6], [1.8, 0.7]]  # 指定中心
    new_point = [1.55, 1.15]
    plt.scatter(new_point[0], new_point[1], s=4200, edgecolors='black', linewidths=1.5, c='white')
    x, y = make_blobs(n_samples=num_points, centers=centers,
                      cluster_std=0.2, random_state=np.random.seed(3))
    index_c0, index_c1, index_c2 = (y == 0), (y == 1), (y == 2)
    c0, c1, c2 = x[index_c0], x[index_c1], x[index_c2]
    plt.scatter(c0[:, 0], c0[:, 1], marker='s', label='class 0', s=50, c='black')
    plt.scatter(c1[:, 0], c1[:, 1], marker='o', label='class 1', s=50, c='black')
    plt.scatter(c2[:, 0], c2[:, 1], marker='*', label='class 2', s=50, c='black')
    plt.scatter(new_point[0], new_point[1], marker='v', s=60, c='black')
    plt.annotate("我是谁？我属于哪个类别？", xy=(1.56, 1.15), fontsize=13,
                 xytext=(0.4, 1.5), arrowprops=dict(arrowstyle="<-", connectionstyle="arc3"))
    plt.xticks([])
    plt.yticks([])
    fm.fontManager.addfont('../data/SimHei.ttf')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来  正常显示中文标签
    plt.legend(fontsize=15)
    plt.tight_layout()  # 调整子图间距
    plt.show()


if __name__ == '__main__':
    visualization()
