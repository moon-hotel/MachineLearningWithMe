import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs


def visualization():
    num_points = 200
    size = 35
    centers = [[1, 1], [2, 2]]  # 指定中心
    x, y = make_blobs(n_samples=num_points, centers=centers, cluster_std=0.2, random_state=np.random.seed(10))
    for i in range(num_points):
        color = 'orange' if y[i] == 0 else 'red'
        mark = 'o' if y[i] == 0 else 's'  # 形状
        plt.scatter(x[i, 0], x[i, 1], c=color, marker=mark, alpha=0.83, cmap='hsv', s=size, )  # alpha 控制透明度
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()  # 调整子图间距
    plt.show()


if __name__ == '__main__':
    visualization()
