import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import pairwise_distances


def get_silhouette_coefficient(X, labels):
    """
    轮廓系数计算
    :param X: shape: [n_samples,n_features]
    :param labels: shape: [n_samples,]
    :return:
    """
    n_clusters = np.unique(labels).shape[0]
    s = []
    for k in range(n_clusters):  # 遍历每一个簇
        index = (labels == k)  # 取对应簇所有样本的索引
        x_in_cluster = X[index]  # 去对应簇中的所有样本
        for sample in x_in_cluster:  # 计算每个样本的轮廓系数
            a = ((sample - x_in_cluster) ** 2).sum(axis=1)
            a = np.sqrt(a).sum() / (len(a) - 1)  # 去掉当前样本点与当前样本点的组合计数
            nearest_cluster_id = None
            min_dist2 = np.inf
            for c in range(n_clusters):  # 寻找距离当前样本点最近的簇
                if k == c:
                    continue
                centroid = X[labels == c].mean(axis=0)
                dist2 = ((sample - centroid) ** 2).sum()
                if dist2 < min_dist2:
                    nearest_cluster_id = c
                    min_dist2 = dist2
            x_nearest_cluster = X[labels == nearest_cluster_id]
            b = ((sample - x_nearest_cluster) ** 2).sum(axis=1)
            b = np.sqrt(b).mean()
            s.append((b - a) / np.max([a, b]))
    return np.mean(s)


def get_calinski_harabasz(X, labels):
    n_samples = X.shape[0]
    n_clusters = np.unique(labels).shape[0]
    betw_disp = 0.  # 所有的簇间距离和
    within_disp = 0.  # 所有的簇内距离和
    global_centroid = np.mean(X, axis=0)  # 全局簇中心
    for k in range(n_clusters):  # 遍历每一个簇
        x_in_cluster = X[labels == k]  # 取当前簇中的所有样本
        centroid = np.mean(x_in_cluster, axis=0)  # 计算当前簇的簇中心
        # 计算所有样本点到其对应簇中心的距离和（平方）
        within_disp += np.sum((x_in_cluster - centroid) ** 2)
        # 计算每个簇中心到全局簇中心的距离和（平方）* 当前簇的样本数
        betw_disp += len(x_in_cluster) * np.sum((centroid - global_centroid) ** 2)

    return (1. if within_disp == 0. else
            betw_disp * (n_samples - n_clusters) /
            (within_disp * (n_clusters - 1.)))


def test_silhouette_score():
    x, y = load_iris(return_X_y=True)
    model = KMeans(n_clusters=3)
    model.fit(x)
    y_pred = model.predict(x)
    print(f"轮廓系数 by sklearn: {silhouette_score(x, y_pred)}")
    print(f"轮廓系数 by ours: {get_silhouette_coefficient(x, y_pred)}")


def test_calinski_harabasz_score():
    x, y = load_iris(return_X_y=True)
    model = KMeans(n_clusters=3)
    model.fit(x)
    y_pred = model.predict(x)
    print(f"方差比 by sklearn: {calinski_harabasz_score(x, y_pred)}")
    print(f"方差比 by ours: {get_calinski_harabasz(x, y_pred)}")


if __name__ == '__main__':
    test_silhouette_score()
    test_calinski_harabasz_score()
