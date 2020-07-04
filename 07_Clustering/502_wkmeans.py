import numpy as np
import random
import math
from sklearn.datasets import make_blobs
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import KMeans


def InitCentroids(X, K):
    n = np.size(X, 0)
    rands_index = np.array(random.sample(range(1, n), K))
    centriod = X[rands_index, :]
    return centriod


def findClosestCentroids(X, w, centroids):
    K = np.size(centroids, 0)
    idx = np.zeros((np.size(X, 0)), dtype=int)
    n = X.shape[0]  # n 表示样本个数
    for i in range(n):
        subs = centroids - X[i, :]
        dimension2 = np.power(subs, 2)
        w_dimension2 = np.multiply(w, dimension2)
        w_distance2 = np.sum(w_dimension2, axis=1)
        if math.isnan(w_distance2.sum()) or math.isinf(w_distance2.sum()):
            w_distance2 = np.zeros(K)
            # print 'the situation that w_distance2 is nan or inf'
        idx[i] = np.where(w_distance2 == w_distance2.min())[0][0]
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


def computeWeight(X, centroid, idx, K, belta):
    n, m = X.shape
    weight = np.zeros((1, m), dtype=float)
    D = np.zeros((1, m), dtype=float)
    for k in range(K):
        index = np.where(idx == k)[0]
        temp = X[index, :]  # 取第k个簇的所有样本
        distance2 = np.power((temp - centroid[k, :]), 2)  # ? by m
        D = D + np.sum(distance2, axis=0)
    e = 1 / float(belta - 1)
    for j in range(m):
        temp = D[0][j] / D[0]
        weight[0][j] = 1 / np.sum((np.power(temp, e)), axis=0)
    return weight


def wkmeans(X, K, belta=7, max_iter=50):
    n, m = X.shape
    r = np.random.rand(1, m)
    w = np.divide(r, r.sum())
    centroids = InitCentroids(X, K)
    idx = None
    for i in range(max_iter):
        idx = findClosestCentroids(X, w, centroids)
        centroids = computeCentroids(X, idx, K)
        w = computeWeight(X, centroids, idx, K, belta)
    return idx


def make_data():
    from sklearn.preprocessing import StandardScaler
    np.random.seed(100)
    centers = [[2.5, 1], [3.8, 1], [3, 2.5]]  # 指定簇中心
    x, y = make_blobs(n_samples=900, centers=centers, cluster_std=0.35, random_state=200)
    noise = np.reshape(np.sin(2 * x[:, 0] * x[:, 1]), [-1, 1])
    x_noise = np.hstack([x, noise])
    ss = StandardScaler()
    x_noise = ss.fit_transform(x_noise)
    return x, y, x_noise


if __name__ == '__main__':
    x, y, x_noise = make_data()
    y_pred = wkmeans(x, 3, belta=3)
    nmi = normalized_mutual_info_score(y, y_pred)
    print("NMI without noise: ", nmi)

    y_pred = wkmeans(x_noise, 3,belta=3)
    nmi = normalized_mutual_info_score(y, y_pred)
    print("NMI with noise : ", nmi)
    #
    #NMI without noise:  0.8641031551829714
    #NMI with noise :  0.852210021893048
    #  可能会出现每次运行结果不一样的情况，这是由于初始化簇中心不同而导致的


