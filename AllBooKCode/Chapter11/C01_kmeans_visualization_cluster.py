from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np


def visualization():
    centers = [[2.2, 1], [3.8, 1], [3, 2.8]]  # 指定簇中心
    x, y = make_blobs(n_samples=800, centers=centers, cluster_std=0.3, random_state=200)
    markers = ['o', 's', 'v']
    for i in range(len(centers)):
        index = np.where(y == i)[0]
        cluster = x[index, :]
        plt.scatter(cluster[:, 0], cluster[:, 1], s=60, marker=markers[i])
    plt.xticks([])
    plt.yticks([])
    plt.show()


if __name__ == '__main__':
    visualization()
