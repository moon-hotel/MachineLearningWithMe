import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import pairwise_distances, adjusted_rand_score
import logging
from copy import deepcopy
import sys
from sklearn.datasets import load_iris, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering


class ClusterNode(object):
    def __init__(self, idx=None, centroid=None):
        """
        :param idx: list,簇中样本点在原始数据集中的索引
        :param centroid: 当前簇的簇中心
        """
        samples = []
        if idx is not None:
            samples = [idx]
        self.samples = samples  # 保存当前节点中存在所有数据集样本在原始数据集中的索引
        self.n_samples = len(self.samples)  # 保存当前节点对应的样本数量
        self.children = {}  # 保存当前节点对应的孩子节点，这个参数也可以不用
        self.centroid = centroid

    def merge(self, node):
        """
        合并节点（簇）
        :param node:
        :return:
        """
        self.samples += node.samples
        for k, v in node.children.items():
            self.children[k] = v
        self.n_samples = len(self.samples)

    def __str__(self):
        """
        打印节点时输出相应信息
        :return:
        """
        return f"样本索引为: {self.samples}  样本数量为: {self.n_samples}"


def single_linkage(X, n_clusters, metric="euclidean"):
    """
    single linkage 实现部分
    :param X: 训练集, shape: [n_samples, n_features]
    :param n_clusters: 簇的数量
    :param metric: 字符串 ['cityblock', 'cosine', 'euclidean', 'l1', 'l2','manhattan'].
    :return:
    """
    cluster_nodes = [ClusterNode(i) for i in range(len(X))]  # 初始化每个样本点为一个簇节点
    old_d = pairwise_distances(X, metric=metric)  # old_d[i][j] 表示第i个簇到第j个簇之间的距离
    # [n_samples,n_samples] 对称矩阵，计算簇与簇之间的距离
    n_merge = 0
    while len(cluster_nodes) > n_clusters:
        n_merge += 1
        merge_dims = None  # 记录需要合并的维度
        all_locations = np.where(np.abs(old_d - np.sort(old_d.ravel())[len(old_d)]) < 1e-6)
        # 用来寻找簇与簇之间最短距离的索引， 由于距离是浮点型，所以通过作差来进行判断
        for locs in zip(*all_locations):
            # 去掉样本重复时，距离为0的错误索引情况，因为old_d[i][i]=0，表示第i个簇与其自身的距离
            if locs[0] == locs[1]:  # 当最小值出现在对角线上时，继续
                continue
            merge_dims = [locs[0], locs[1]]  # 否则得到最小距离对应的索引
            break
        # logging.debug(f"第{n_merge}次合并前 old D:\n {old_d}")
        logging.debug(f"第{n_merge}次合并前 簇的个数为: {len(cluster_nodes)}")
        # 遍历所有需要合并的节点，并将需要被合并的簇从列表中删除
        del_nodes = [cluster_nodes.pop(dim) for dim in merge_dims[::-1]]
        del_nodes[0].merge(del_nodes[1])  #
        del del_nodes[1]
        logging.debug(f"第{n_merge}次合并前 合并节点的信息:({del_nodes[0]})")
        cluster_nodes.insert(0, del_nodes[0])  # 将合并后的节点插入到最前面的位置
        logging.debug(f"第{n_merge}次合并后 簇的个数为: {len(cluster_nodes)}")
        new_d = deepcopy(old_d)
        # 拷贝d，并从new_d中删除合并的两个簇对应的行和列
        new_d = np.delete(np.delete(new_d, merge_dims, axis=0), merge_dims, axis=1)
        new_d = np.pad(new_d, [1, 0])  # 在new_d最第一行和最第一列padding 0
        old_d_dims = [i for i in range(len(old_d)) if i not in merge_dims]  # 得到上一个d去掉合并维度后剩下的维度
        new_d_dims = [i for i in range(1, len(new_d))]  # 第一个位置是新插入的簇节点，所以从1开始
        logging.debug(f"第{n_merge}次合并后 old_d_dims: {old_d_dims}")
        logging.debug(f"第{n_merge}次合并后 merge_dims: {merge_dims}")
        for i, j in zip(new_d_dims, old_d_dims):
            value = np.inf
            for k in merge_dims:
                value = np.min([old_d[k][j], value])  # 寻找最小距离
            new_d[0][i] = new_d[i][0] = value  # 更新 new_d
        old_d = new_d
        logging.debug(f"第{n_merge}次合并后 new D:\n {new_d}")
        logging.debug(f" ======= 第{n_merge}次合并结束 "
                      f" 此时各个簇中样本分布情况为{[node.n_samples for node in cluster_nodes]}=======")
    return cluster_nodes


def ward_linkage(X, n_clusters, metric="euclidean"):
    """
    single linkage 实现部分
    :param X: 训练集, shape: [n_samples, n_features]
    :param n_clusters: 簇的数量
    :param metric: 字符串 ['cityblock', 'cosine', 'euclidean', 'l1', 'l2','manhattan'].
    :return:
    """
    # 初始化每个样本点为一个簇节点，并同时初始化该簇对应的簇中心
    cluster_nodes = [ClusterNode(i, X[i]) for i in range(len(X))]
    n_merge = 0
    while len(cluster_nodes) > n_clusters:
        centroids, n_samples = [], []
        for node in cluster_nodes:
            centroids.append(node.centroid)  # 得到所有簇的簇中心
            n_samples.append(node.n_samples)  # 得到每个簇对应的样本数量
        n_samples = np.array(n_samples)
        weight = (n_samples[:, None] * n_samples) / (n_samples[:, None] + n_samples)  # 计算权重
        d = weight * pairwise_distances(centroids, metric=metric)
        # [n_samples,n_samples] 对称矩阵，计算簇与簇之间的距离
        n_merge += 1
        merge_dims = None
        all_locations = np.where(np.abs(d - np.sort(d.ravel())[len(d)]) < 1e-6)  # 顺序
        # 用来寻找簇与簇之间最短距离的索引， 由于距离是浮点型，所以通过作差来进行判断
        # 注意，这里np.sort()排序后的结果一定要为顺序
        for locs in zip(*all_locations):  # 去掉样本重复时，距离为0的错误索引情况
            if locs[0] == locs[1]:  # 去掉矩阵d中 对角线上的情况
                continue
            merge_dims = [locs[0], locs[1]]
            break

        logging.debug(f"第{n_merge}次合并前 D:\n {d}")
        logging.debug(f"第{n_merge}次合并前 当前簇个数为: {len(cluster_nodes)}")

        del_nodes = [cluster_nodes.pop(dim) for dim in merge_dims[::-1]]  # 遍历所有需要合并的节点
        del_nodes[0].merge(del_nodes[1])  #
        del_nodes[0].centroid = np.mean(X[del_nodes[0].samples], axis=0)
        logging.debug(f"第{n_merge}次合并前 合并节点的信息:({del_nodes[0]})")
        cluster_nodes.insert(0, del_nodes[0])  # 将合并后的节点插入到最前面的位置
        logging.debug(f" ======= 第{n_merge}次合并结束 "
                      f" 此时各个簇中样本分布情况为{[node.n_samples for node in cluster_nodes]}=======")
    return cluster_nodes


class HierarchicalClustering(object):
    """
    Parameters:
        n_clusters: 簇数量
        linkage: 指定层次聚类的策略
        metric: 指定距离计算方式['cityblock', 'cosine', 'euclidean', 'l1', 'l2','manhattan'].
    Attributes:
        labels_:  shape (n_samples,), 聚类后的簇标签
    """

    def __init__(self, n_clusters=3, linkage="single", metric='euclidean'):
        self.linkage = linkage
        self.metric = metric
        self.n_clusters = n_clusters

    def fit(self, X):
        cluster_nodes = None
        if self.linkage == "single":
            cluster_nodes = single_linkage(X, self.n_clusters, self.metric)
        elif self.linkage == "ward":
            cluster_nodes = ward_linkage(X, self.n_clusters, self.metric)
        else:
            raise ValueError(f"self.linkage == {self.linkage} 不存在该方法！")
        labels_ = [-1] * len(X)
        for cluster_id, node in enumerate(cluster_nodes):
            for sample in node.samples:
                labels_[sample] = cluster_id
        self.labels_ = labels_


def test_single():
    n_clusters = 2
    X, y = make_moons(n_samples=500, noise=0.05, random_state=2020)

    # n_clusters = 3
    # X, y = load_iris(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    my_single = HierarchicalClustering(n_clusters=n_clusters)
    my_single.fit(X)
    logging.info(f"HierarchicalClustering 聚类结果兰德系数为: {adjusted_rand_score(y, my_single.labels_)}")
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage="single")
    model.fit(X)
    logging.info(f"AgglomerativeClustering 聚类结果兰德系数为: {adjusted_rand_score(y, model.labels_)}")

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title("True Distribution")
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.subplot(1, 2, 2)
    plt.title("Clustered by MySingle")
    plt.scatter(X[:, 0], X[:, 1], c=my_single.labels_)
    plt.tight_layout()
    plt.show()


def test_ward():
    # n_clusters = 2
    # X, y = make_moons(n_samples=500, noise=0.05, random_state=2020)

    n_clusters = 3
    X, y = load_iris(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    my_ward = HierarchicalClustering(n_clusters=n_clusters, linkage="ward")
    my_ward.fit(X)
    logging.info(f"HierarchicalClustering 聚类结果兰德系数为: {adjusted_rand_score(y, my_ward.labels_)}")
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    model.fit(X)
    logging.info(f"AgglomerativeClustering 聚类结果兰德系数为: {adjusted_rand_score(y, model.labels_)}")

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title("True Distribution")
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.subplot(1, 2, 2)
    plt.title("Clustered by MyWard")
    plt.scatter(X[:, 0], X[:, 1], c=my_ward.labels_)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    formatter = '[%(asctime)s] - %(levelname)s: %(message)s'
    logging.basicConfig(level=logging.INFO,  # 如果需要查看详细信息可将该参数改为logging.DEBUG
                        format=formatter,  # 关于Logging模块的详细使用可参加文章https://www.ylkz.life/tools/p10958151/
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[logging.StreamHandler(sys.stdout)])
    test_single()
    test_ward()
