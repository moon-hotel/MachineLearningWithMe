import numpy as np
import random
from sklearn.datasets import load_iris
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import KMeans

random.seed(12)


def load_data():
    data = load_iris()
    x, y = data.data, data.target
    return x, y


def InitCentroids(X, K):
    n = X.shape[0]
    rands_index = np.array(random.sample(range(0, n), K))
    centriod = X[rands_index, :]
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
        centriod[k, :] = s / index.shape[0]
    return centriod


def kmeans(X, K, max_iter=200):
    centroids = InitCentroids(X, K)
    idx = None
    for i in range(max_iter):
        idx = findClostestCentroids(X, centroids)
        centroids = computeCentroids(X, idx, K)
    return idx


if __name__ == '__main__':
    x, y = load_data()
    K = len(np.unique(y))
    y_pred = kmeans(x, K)
    nmi = normalized_mutual_info_score(y, y_pred)
    print("NMI by ours: ", nmi)

    model = KMeans(n_clusters=K)
    model.fit(x)
    y_pred = model.predict(x)
    nmi = normalized_mutual_info_score(y, y_pred)
    print("NMI by sklearn: ", nmi)

    # NMI by ours:  0.7581756800057784
    # NMI by sklearn:  0.7581756800057784
