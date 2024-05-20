import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


def InitCentroids(X, K, idx):
    np.random.seed(17)
    r_cluster = np.random.permutation(len(idx))[0]
    # 确定所有簇中心都初始化到哪个簇
    r_cluster = idx[r_cluster]
    r_idx = np.where(idx == r_cluster)[0]
    x = X[r_idx, :]

    n = x.shape[0]
    rands_index = np.random.permutation(n)[:K]
    centriod = x[rands_index, :]
    return centriod


def findClostestCentroids(X, centroid):
    n = X.shape[0]  # n 表示样本个数
    idx = np.zeros(n, dtype=int)
    for i in range(n):
        subs = centroid - X[i, :]
        dimension2 = np.power(subs, 2)
        dimension_s = np.sum(dimension2, axis=1)  # sum of each row
        idx[i] = np.where(dimension_s == dimension_s.min())[0][0]
    return idx


def computeCentroids(X, idx, K):
    n, m = X.shape
    centriod = np.zeros((K, m), dtype=float)
    for k in range(K):
        index = np.where(idx == k)[0]  # 一个簇一个簇的分开来计算
        temp = X[index, :]  # ? by m # 每次先取出一个簇中的所有样本
        s = np.sum(temp, axis=0)
        centriod[k, :] = s / np.size(index)
    return centriod


def make_data():
    centers = [[2.2, 1], [3.8, 1], [3, 2.8]]  # 指定簇中心
    x, y = make_blobs(n_samples=900, centers=centers, cluster_std=0.35, random_state=200)
    return x, y


def kmeans(X, idx, K):
    plt.figure(figsize=(12, 4), dpi=80)
    centroids = InitCentroids(X, K, idx)
    fig_idx = 1
    row, col = 2, 3
    step = 10
    for i in range(row * col * step):
        if i % step == 0:
            index_c0, index_c1, index_c2 = (idx == 0), (idx == 1), (idx == 2)
            c0, c1, c2 = X[index_c0], X[index_c1], X[index_c2]
            plt.subplot(row, col, fig_idx)
            plt.scatter(c0[:, 0], c0[:, 1])
            plt.scatter(c1[:, 0], c1[:, 1])
            plt.scatter(c2[:, 0], c2[:, 1])
            plt.scatter(centroids[0, 0], centroids[0, 1], c='black', s=50)
            plt.scatter(centroids[1, 0], centroids[1, 1], c='black', s=50)
            plt.scatter(centroids[2, 0], centroids[2, 1], c='black', s=50)
            plt.xlabel("iter = {}".format(i), fontsize=15)
            fig_idx += 1
            plt.xticks([])
            plt.yticks([])
        idx = findClostestCentroids(X, centroids)
        centroids = computeCentroids(X, idx, K)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    x, y = make_data()
    kmeans(x, y, 3)
