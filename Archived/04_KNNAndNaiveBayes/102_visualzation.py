import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs


def visualization():
    num_points = 300
    plt.figure(figsize=(10,5),dpi=120)
    plt.subplot(1, 2, 1)
    centers = [[1, 1], [1.8, 1.6], [1.8, 0.7]]  # 指定中心
    new_point=[1.55,1.15]
    plt.scatter(new_point[0], new_point[1], s=1000, edgecolors='r', linewidths=1.5, c='white')
    x, y = make_blobs(n_samples=num_points, centers=centers, cluster_std=0.2, random_state=np.random.seed(3))
    index_c0, index_c1, index_c2 = (y == 0), (y == 1), (y == 2)
    c0, c1, c2 = x[index_c0], x[index_c1], x[index_c2]
    plt.scatter(c0[:, 0], c0[:, 1], marker='s', label='class 0')
    plt.scatter(c1[:, 0], c1[:, 1], marker='o', label='class 1')
    plt.scatter(c2[:, 0], c2[:, 1], marker='v', label='class 2')
    plt.scatter(new_point[0], new_point[1], s=30,c='black')
    plt.xticks([])
    plt.yticks([])
    plt.legend(fontsize=15)
    plt.subplot(1, 2, 2)
    plt.scatter(new_point[0], new_point[1], s=6200, edgecolors='r', linewidths=1.5, c='white')
    x, y = make_blobs(n_samples=num_points, centers=centers, cluster_std=0.2, random_state=np.random.seed(3))
    index_c0, index_c1, index_c2 = (y == 0), (y == 1), (y == 2)
    c0, c1, c2 = x[index_c0], x[index_c1], x[index_c2]
    plt.scatter(c0[:, 0], c0[:, 1], marker='s', label='class 0')
    plt.scatter(c1[:, 0], c1[:, 1], marker='o', label='class 1')
    plt.scatter(c2[:, 0], c2[:, 1], marker='v', label='class 2')
    plt.scatter(new_point[0], new_point[1], s=30,c='black')
    plt.xticks([])
    plt.yticks([])

    plt.rcParams['font.sans-serif'] = ['SimSun']  # 用来  正常显示中文标签
    plt.legend(fontsize=15)
    plt.tight_layout()  # 调整子图间距
    plt.show()


if __name__ == '__main__':
    visualization()
