from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import sys
import logging
import time

sys.path.append('../')
from Chapter05.C05_knn_imp import MyKDTree


def my_dbscan_inner(is_core, neighborhoods, labels):
    """
    深度优先遍历，判断每个样本的所属簇类别
    :param is_core:  只含0和1 ，shape: [n_samples,], is_core[i] == 1 表示第i个样本为核心样本
    :param neighborhoods: list, neighborhoods[i] 表示第i个样本以r为半径周围存在样本的索引
    :param labels: shape: [n_samples,], labels[i]表示第i个样本所属的簇编号，初始情况全部为-1, 即没有被访问过
    :return:
    """
    label_num = 0  # 用来记录簇的编号
    stack = []  # 深度优先遍历时用的栈

    for i in range(labels.shape[0]):
        if labels[i] != -1 or not is_core[i]:  # labels[i] != -1表示该样本的已经访问过
            continue  # is_core[i] == 0 表示第i个样本不是核心样本, 即
            # 如果该样本已经访问过 或 者不是核心样本则跳过，只有没访问过 且为 核心样本才继续往下
        idx = i
        while True:
            if labels[idx] == -1:  # 如果第idx个样本没有被访问过
                labels[idx] = label_num  # 为当前样本赋予一个簇类别
                if is_core[idx]:  # 如果当前样本为核心样本
                    neighb = neighborhoods[idx]  # 取以第idx个核心样本为圆心，周围半径r内的所有样本点
                    for j in range(neighb.shape[0]):  # 遍历第idx个核心样本周围的所有样本
                        v = neighb[j]
                        if labels[v] == -1:  # 查看第idx个核心样本周围的每个样本点是否被访问
                            stack.append(v)  # 如果没访问则加入栈中
            if len(stack) == 0:  # 如果栈为空，则表示当前簇中所有样本都访问完毕
                break
            idx = stack.pop()  # 返回最后一个元素并将其从栈中删除
        label_num += 1


class MyNearestNeighbors(MyKDTree):
    """
    利用KD树来查找当前样本点在以r为半径的范围内的所有样本
    """

    def __init__(self, X, p, r=0.5):
        """
        :param X: 样本点
        :param p: p = 2时为欧式距离
        :param r: 半径
        """
        super(MyNearestNeighbors, self).__init__(points=X, p=p)
        self.r = r

    def radius_neighbors(self, points):
        """
        分别查找points中每个样本周围半径为self.r以内的样本点（包括自身）
        :param points: np.array()   形状为[n,m]
        :return:找到的样本点，及对应的索引
        """
        result_points = []
        result_ind = []
        for point in points:
            radius_nodes = self.query_radius_single(point)
            tmp_points = []
            tmp_ind = []
            for node in radius_nodes:
                tmp_points.append(node.data)
                tmp_ind.append(int(node.index))
            result_points.append(np.array(tmp_points))
            result_ind.append(np.array(tmp_ind, dtype=np.int64))
            logging.debug("搜索下一个样本点……")
        return result_points, result_ind

    def query_radius_single(self, point):
        """
        寻找距离样本点point周围半径为self.r以内的样本点
        :param point: np.array()   形状为[m,]
        :return:
        """
        query_radius_nodes = []
        visited = []

        def radius_node(point, curr_node, order=0):
            nonlocal query_radius_nodes
            logging.debug(f"    query_radius 搜索当前访问节点为：{curr_node}")
            if curr_node is None:
                return None
            visited.append(curr_node)
            if self.distance(point, curr_node.data, self.p) <= self.r:
                query_radius_nodes = self.append(query_radius_nodes, curr_node, point)
            cmp_dim = order % self.dim
            if point[cmp_dim] < curr_node.data[cmp_dim]:
                radius_node(point, curr_node.left_child, order + 1)
            else:
                radius_node(point, curr_node.right_child, order + 1)
            if np.abs(curr_node.data[cmp_dim] - point[cmp_dim]) < self.r:
                child = curr_node.left_child if curr_node.left_child \
                                                not in visited else curr_node.right_child
                radius_node(point, child, order + 1)

        radius_node(point, self.root, 0)
        return query_radius_nodes


class MyDBSCAN(object):
    def __init__(self, eps=0.5, p=2, min_samples=5):
        self.eps = eps
        self.p = p
        self.min_samples = min_samples  # 成为核心样本周围样本的最小数量

    def fit(self, X):
        neighbors_model = MyNearestNeighbors(X, p=self.p, r=self.eps)
        time1 = time.time()
        _, neighborhoods = neighbors_model.radius_neighbors(X)
        time2 = time.time()
        logging.info(f"计算每个样本点周围的样本花费时间为：{time2 - time1}s")
        # 得到每个样本点在以self.eps为半径的区域内存在样本点的个数
        n_neighbors = np.array([len(neighbors) for neighbors in neighborhoods])
        labels = np.full(X.shape[0], -1, dtype=np.intp)

        core_samples = np.asarray(n_neighbors >= self.min_samples, dtype=np.uint8)
        my_dbscan_inner(core_samples, neighborhoods, labels)
        time3 = time.time()
        logging.info(f"对每个样本点进行簇划分花费的时间为{time3 - time2}s")
        self.labels_ = labels


def test_moon():
    X, y = make_moons(n_samples=700, noise=0.05, random_state=2020)
    model = KMeans(n_clusters=2)
    model.fit(X)
    y_pred = model.predict(X)
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.7)
    plt.title("True Distribution")
    plt.subplot(1, 2, 2)
    plt.title("Clustered by KMeans")
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, alpha=0.7)
    plt.tight_layout()
    plt.show()


def test_query_radius():
    import numpy as np
    from sklearn.neighbors import NearestNeighbors
    samples = np.array([[0., 0.], [4, 5], [0., 1], [1., 1],
                        [3, 6], [4, 9], [7, 2], [10, 7], [8, 8], [9, 6]])
    query = np.array([[1., 2], [1, 1], [4, 3], [7, 5], [6, 6]])

    r = 1.6
    neigh = NearestNeighbors(radius=r, leaf_size=1)
    neigh.fit(samples)
    neighborhoods = neigh.radius_neighbors(query, return_distance=False)
    logging.info(f" ====== 查找样本点周围r={r}内的样本，NearestNeighbors 运行结果：")
    logging.info(neighborhoods)
    print(type(neighborhoods))
    model = MyNearestNeighbors(samples, p=2, r=r)
    _, neighborhoods = model.radius_neighbors(query)
    logging.info(f" ====== 查找样本点周围r={r}内的样本，MyNearestNeighbors 运行结果：")
    logging.info(neighborhoods)


def test_circle_dbscan():
    plt.figure(figsize=(8, 4))
    X, labels_true = make_circles(n_samples=700, noise=0.05, random_state=2022, factor=0.5)
    X = StandardScaler().fit_transform(X)
    db = DBSCAN(eps=0.3, min_samples=10)
    db.fit(X)
    logging.info(f"DBSCAN 聚类结果兰德系数为: {adjusted_rand_score(labels_true, db.labels_)}")
    my_db = MyDBSCAN(eps=0.3, min_samples=10)
    my_db.fit(X)
    logging.info(f"MyDBSCAN 聚类结果兰德系数为: {adjusted_rand_score(labels_true, my_db.labels_)}")

    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=db.labels_, alpha=0.7)
    plt.title("Clustered by DBSCAN")
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=my_db.labels_, alpha=0.7)
    plt.title("Clustered by MyDBSCAN")
    plt.tight_layout()
    plt.show()


def test_moon_dbscan():
    X, y = make_moons(n_samples=700, noise=0.05, random_state=2020)
    X = StandardScaler().fit_transform(X)
    model = KMeans(n_clusters=2)
    model.fit(X)
    y_pred = model.predict(X)
    my_db = MyDBSCAN(eps=0.3, min_samples=10)
    my_db.fit(X)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title("Clustered by KMeans")
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, alpha=0.7)
    plt.subplot(1, 2, 2)
    plt.title("Clustered by MyDBSCAN")
    plt.scatter(X[:, 0], X[:, 1], c=my_db.labels_, alpha=0.7)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    formatter = '[%(asctime)s] - %(levelname)s: %(message)s'
    logging.basicConfig(level=logging.INFO,  # 如果需要查看详细信息可将该参数改为logging.DEBUG
                        format=formatter,  # 关于Logging模块的详细使用可参加文章https://www.ylkz.life/tools/p10958151/
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[logging.StreamHandler(sys.stdout)])
    # test_moon()
    test_query_radius()
    test_circle_dbscan()
    # test_moon_dbscan()
