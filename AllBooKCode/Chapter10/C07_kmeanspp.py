import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics.cluster import normalized_mutual_info_score
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def make_data():
    centers = [[2.2, 1], [3.8, 1], [3, 2.8]]  # 指定簇中心
    x, y = make_blobs(n_samples=900, centers=centers, cluster_std=0.35, random_state=200)
    return x, y


def InitialCentroid(x, K):
    c0_idx = int(np.random.uniform(0, len(x)))
    centroid = x[c0_idx].reshape(1, -1)  # 选择第一个簇中心
    k, n = 1, x.shape[0]
    while k < K:
        d2 = []
        for i in range(n):
            subs = centroid - x[i, :]
            dimension2 = np.power(subs, 2)
            dimension_s = np.sum(dimension2, axis=1)  # sum of each row
            d2.append(np.min(dimension_s))
        new_c_idx = np.argmax(d2)
        centroid = np.vstack([centroid, x[new_c_idx]])
        k += 1
    return centroid


def findClostestCentroids(X, centroid):
    idx = np.zeros((np.size(X, 0)), dtype=int)
    n = X.shape[0]  # n 表示样本个数
    for i in range(n):
        subs = centroid - X[i, :]
        dimension2 = np.power(subs, 2)
        dimension_s = np.sum(dimension2, axis=1)  # sum of each row
        dimension_s = np.nan_to_num(dimension_s)
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


def kmeanspp(X, K, max_iter=200):
    centroids = InitialCentroid(X, K)
    idx = None
    for i in range(max_iter):
        idx = findClostestCentroids(X, centroids)
        centroids = computeCentroids(X, idx, K)
    return idx, centroids


def kmeanspp_visual(X, idx, K):
    plt.figure(figsize=(12, 4), dpi=80)
    centroids = InitialCentroid(X, K)
    fig_idx = 1
    row, col = 2, 3
    step = 1
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
            plt.xlabel("iter = {}".format(fig_idx - 1), fontsize=15)
            fig_idx += 1
            plt.xticks([])
            plt.yticks([])
        idx = findClostestCentroids(X, centroids)
        centroids = computeCentroids(X, idx, K)
    plt.tight_layout()
    plt.show()
    return idx


if __name__ == '__main__':
    x, y = make_data()
    K = len(np.unique(y))
    y_pred, centroids = kmeanspp(x, K)
    # y_pred = kmeanspp_visual(x, y, K)
    nmi = normalized_mutual_info_score(y, y_pred)
    print("NMI by ours: ", nmi)
    print("centroids: ", centroids)

    model = KMeans(n_clusters=K, init='k-means++')
    model.fit(x)
    y_pred = model.predict(x)
    nmi = normalized_mutual_info_score(y, y_pred)
    print("NMI by sklearn: ", nmi)
    print("centroids: ", model.cluster_centers_)
