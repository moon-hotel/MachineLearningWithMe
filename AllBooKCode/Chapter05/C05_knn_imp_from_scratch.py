"""
文件名: AllBooKCode/Chapter05/C05_knn_imp_from_scratch.py
作 者: @空字符
B 站: @月来客栈Moon https://space.bilibili.com/392219165
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
油 管: @月来客栈
小红书: @月来客栈
公众号: @月来客栈
代码仓库: https://github.com/moon-hotel/MachineLearningWithMe
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KDTree
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import numpy as np
import logging
import sys


def load_data():
    x, y = load_iris(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=0.3, random_state=20)
    return x_train, x_test, y_train, y_test


class Node(object):
    """
    定义KD树中的节点信息
    """

    def __init__(self, data=None, index=-1):
        self.data = data
        self.left_child = None
        self.right_child = None
        self.index = index

    def __str__(self):
        """
        打印节点信息
        :return:
        """
        return f"data({self.data}),index({int(self.index)})"


def distance(p1, p2, p=2):
    """

    :param p1:
    :param p2:
    :param p:  当p=2时为欧式距离
    :return:
    """
    return np.sum((p1 - p2) ** p) ** (1 / p)


class MyKDTree(object):
    """
    定义MyKDTree
    Parameters:
        points: 构造KD树用到的样本点，必须为np.array()类型，形状为 [n,m]
        p: p=2时为欧式距离
    """

    def __init__(self, points, p=2):
        self.root = None
        self.p = p
        self.dim = points.shape[1]
        self.distance = distance
        # 在原始样本的最后一列加入一列索引，表示每个样本点的序号
        points = np.hstack(([points, np.arange(0, len(points)).reshape(-1, 1)]))
        self.insert(points, order=0)  # 递归构建KD树

    def is_empty(self):
        return not self.root

    def insert(self, data, order=0):
        """
        在KD树中插入节点
        :param data: 必须为np.array()类型，形状为 [n,m]
        :param order: 用来判断比较的维度
        :return:
        """
        if len(data) < 1:
            return
        data = sorted(data, key=lambda x: x[order % self.dim])  # 按某个维度进行排序
        logging.info(f"当前待划分样本点：{[item.tolist() for item in data]}")
        idx = len(data) // 2
        node = Node(data[idx][:-1], data[idx][-1])
        logging.info(f"父节点：{data[idx]}")
        left_data = data[:idx]
        logging.info(f"左子树: {[item.tolist() for item in left_data]}")
        right_data = data[idx + 1:]
        logging.info(f"右子树: {[item.tolist() for item in right_data]}")
        logging.info("============")
        if self.is_empty():
            self.root = node  # 整个KD树的根节点
        node.left_child = self.insert(left_data, order + 1)  # 递归构建左子树
        node.right_child = self.insert(right_data, order + 1)  # 递归构建右子树
        return node

    def level_order(self):
        """
        层次遍历
        :return:
        """
        root = self.root
        if not root:
            return []
        queue = [root]
        res = []
        while queue:
            tmp = []
            for i in range(len(queue)):
                node = queue.pop(0)
                tmp.append(node)
                if node.left_child:
                    queue.append(node.left_child)
                if node.right_child:
                    queue.append(node.right_child)
            res.append(tmp)
        logging.info("\n层次遍历结果为：")
        for i, r in enumerate(res):
            logging.info(f"第{i + 1}层的节点为：")
            for node in r:
                logging.info(f"<p({node.data}), idx({int(node.index)})>")
            logging.info("\n")

    def nearest_search(self, point):
        """
        最近邻搜索
        :param point: 必须为np.array()类型
        :return:
        """
        best_node = None
        best_dist = np.inf
        visited = []  # 用来记录哪些节点被访问过
        point = point.reshape(-1)

        def nearest_node_search(point, curr_node, order=0):
            """
            :param point:  shape: (n,)
            :param curr_node: data: shape (n,)
            :param order:
            :return:
            """
            nonlocal best_node, best_dist, visited  # 声明这三个变量不是局部变量
            logging.info(f"当前访问节点为：{curr_node}")
            if curr_node is None:
                logging.info(f"结束本次递归")
                return None
            visited.append(curr_node)
            dist = self.distance(curr_node.data, point, self.p)
            logging.info(f"当前访问节点到被搜索点的距离为：{round(dist, 3)}")
            logging.info(f"【上次】:全局最佳距离为：{round(best_dist, 3)}, 全局最佳点为：{best_node}")
            if dist < best_dist:
                best_dist = dist
                best_node = curr_node
            logging.info(f"【本次】:全局最佳距离为：{round(best_dist, 3)}, 全局最佳点为：{best_node}\n")
            cmp_dim = order % self.dim
            if point[cmp_dim] < curr_node.data[cmp_dim]:
                logging.info(f"访问当前节点{curr_node}的左孩子")
                nearest_node_search(point, curr_node.left_child, order + 1)
            else:
                logging.info(f"访问当前节点{curr_node}的右孩子")
                nearest_node_search(point, curr_node.right_child, order + 1)
            logging.info(f"回到上一层递归，当前访问节点为: {curr_node}，开始判断步骤(6)")
            if np.abs(curr_node.data[cmp_dim] - point[cmp_dim]) < best_dist:
                logging.info(f"** {curr_node.data}满足条件被搜索点到当前节点划分维度的距离小于全局最短距离，即 "
                             f"|{curr_node.data[cmp_dim]} - {point[cmp_dim]}| < {round(best_dist, 3)} **")
                child = curr_node.left_child if curr_node.left_child not in visited else curr_node.right_child
                tmp = '左' if curr_node.left_child not in visited else '右'
                logging.info(f"访问当前节点{curr_node}的{tmp}孩子")
                nearest_node_search(point, child, order + 1)

        nearest_node_search(point, self.root, 0)
        return best_node, best_dist

    def append(self, k_nearest_nodes, curr_node, point):
        """
        将当前节点加入到k_nearest_nodes中并按每个节点到被搜索点的距离升序排列
        :param k_nearest_nodes: list，用来保存到目前位置距离被搜索点最近的K个点
        :param curr_node:  Node()类型，当前访问的节点
        :param point: 被搜索点，np.array()类型，形状为(m,)
        :return:
        """
        k_nearest_nodes.append(curr_node)
        k_nearest_nodes = sorted(k_nearest_nodes,
                                 key=lambda x: self.distance(x.data, point, self.p))
        logging.info(f"\t\t当前K近邻中的节点为（已按距离升序排序）：")
        for item in k_nearest_nodes:
            logging.info(f"\t\t\t{item}", )
        logging.info("\n")
        return k_nearest_nodes

    def k_nearest_search(self, points, k):
        """
        分别查找points中每个样本距其最近的k个样本点
        :param points: np.array()   形状为[n,m]
        :param k:
        :return:
        """
        result_points = []
        result_ind = []
        for point in points:
            k_nodes = self._k_nearest_search(point, k)
            tmp_points = []
            tmp_ind = []
            for node in k_nodes:
                tmp_points.append(node.data)
                tmp_ind.append(int(node.index))
            result_points.append(tmp_points)
            result_ind.append(tmp_ind)
        return np.array(result_points), np.array(result_ind)

    def _k_nearest_search(self, point, k):
        """
        寻找距离样本点point最近的k个样本点
        :param point: np.array()   形状为[m,]
        :param k:
        :return:
        """
        k_nearest_nodes = []
        visited = []
        n = 0
        logging.info(f"\n\n====== 正在查找离样本点{point}最近的{k}个样本点！")

        def k_nearest_node_search(point, curr_node, order=0):
            nonlocal k_nearest_nodes, n
            logging.info(f"    K近邻搜索当前访问节点为：{curr_node}")
            if curr_node is None:
                return None
            visited.append(curr_node)
            if n < k:  # 如果当前还没找到k个点，则直接进行保存
                n += 1
                k_nearest_nodes = self.append(k_nearest_nodes, curr_node, point)
            else:  # 已经找到k个局部最优点，开始进行筛选
                dist = (self.distance(curr_node.data, point, self.p)
                        < self.distance(point, k_nearest_nodes[-1].data, self.p))
                if dist:
                    k_nearest_nodes.pop()  # 移除最后一个
                    k_nearest_nodes = self.append(k_nearest_nodes, curr_node, point)  # 加入新的点并进行排序
            cmp_dim = order % self.dim
            if point[cmp_dim] < curr_node.data[cmp_dim]:
                k_nearest_node_search(point, curr_node.left_child, order + 1)
            else:
                k_nearest_node_search(point, curr_node.right_child, order + 1)
            if n < k or np.abs(curr_node.data[cmp_dim] - point[cmp_dim]) < \
                    self.distance(point, k_nearest_nodes[-1].data, self.p):
                logging.info(
                    f"** {curr_node.data}被搜索点到当前节点划分维度的距离小于列表中最后一个元素到被搜索点的距离，即 "
                    f"|{curr_node.data[cmp_dim]} - {point[cmp_dim]}| < "
                    f"{round(self.distance(point, k_nearest_nodes[-1].data, self.p), 3)} **")
                child = curr_node.left_child if curr_node.left_child not in visited else curr_node.right_child
                k_nearest_node_search(point, child, order + 1)

        k_nearest_node_search(point, self.root, 0)
        return k_nearest_nodes


class MyKNN():
    def __init__(self, n_neighbors, p=2):
        """
        :param n_neighbors:
        :param p: 当p=2时表示欧氏距离
        """
        self.n_neighbors = n_neighbors
        self.p = p

    def fit(self, x, y):
        self._y = y
        self.kd_tree = MyKDTree(x, self.p)
        return self

    @staticmethod
    def get_pred_labels(query_label):
        """
        根据query_label返回每个样本对应的标签
        :param query_label: 二维数组， query_label[i] 表示离第i个样本最近的k个样本点对应的正确标签
        :return:
        """
        y_pred = [0] * len(query_label)
        for i, label in enumerate(query_label):
            max_freq = 0
            count_dict = {}
            for l in label:
                count_dict[l] = count_dict.setdefault(l, 0) + 1
                if count_dict[l] > max_freq:
                    max_freq = count_dict[l]
                    y_pred[i] = l
        return np.array(y_pred)

    def predict(self, x):
        k_best_nodes, ind = self.kd_tree.k_nearest_search(x, k=self.n_neighbors)
        query_label = self._y[ind]
        y_pred = self.get_pred_labels(query_label)
        return y_pred


def test_kd_tree_build(points):
    logging.info("MyKDTree 运行结果：")
    logging.info("构建KD树")
    tree = MyKDTree(points)
    logging.info("构建结束")
    logging.info("MyKDTree 层次遍历结果：")
    tree.level_order()


def test_kd_nearest_search(points, q=None):
    tree = MyKDTree(points)
    best_node, best_dist = tree.nearest_search(q)
    logging.info("MyKDTree 运行结果：")
    logging.info(f"离样本点{q}最近的节点是：{best_node},距离为：{round(best_dist, 3)}")

    kd_tree = KDTree(points)
    dist, ind = kd_tree.query(q, k=1)
    logging.info("sklearn KDTree 运行结果：")
    logging.info(f"离样本点{q}最近的节点是：{points[ind]},距离为：{dist}")


def test_kd_k_nearest_search():
    points = np.array(
        [[5, 7], [3, 8], [6, 3], [8, 5], [15, 6.], [10, 4], [12, 13], [9, 10], [11, 14]])
    tree = MyKDTree(points)
    p = np.array([[8.9, 4]])
    k_best_nodes, ind = tree.k_nearest_search(p, k=3)
    logging.info("MyKDTree 运行结果：")
    logging.info(f"\n{k_best_nodes}")
    logging.info(ind)

    logging.info("sklearn KDTree 运行结果：")
    kd_tree = KDTree(points)
    dist, ind = kd_tree.query(p, k=3)
    logging.info(f"\n{points[ind]}", )
    logging.info(ind)


def test_kd_tree_build_in_book():
    """
    构建书中的插图
    :return:
    """
    points = np.array(
        [[5, 7], [3, 8], [6, 3], [8, 5], [15, 6.], [10, 4], [12, 13], [9, 10], [11, 14]])
    test_kd_tree_build(points)


def test_kd_tree_nearest_search_in_book():
    points = np.array(
        [[5, 7], [3, 8], [6, 3], [8, 5], [15, 6.], [10, 4], [12, 13], [9, 10], [11, 14]])
    test_kd_nearest_search(points, q=np.array([[8.9, 4]]))


if __name__ == '__main__':
    formatter = '[%(asctime)s] - %(levelname)s: %(message)s'
    logging.basicConfig(level=logging.DEBUG,  # 如果需要查看详细信息可将该参数改为logging.DEBUG
                        format=formatter,  # 关于Logging模块的详细使用可参加文章https://www.ylkz.life/tools/p10958151/
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[logging.StreamHandler(sys.stdout)])
    # test_kd_tree_build_in_book()  # 构建书中的示例kd树
    test_kd_tree_nearest_search_in_book()  # 书中的最近邻搜索
    # test_kd_k_nearest_search() # 书中的K近邻搜索示例

    # 测试KNN
    # x_train, x_test, y_train, y_test = load_data()
    # k = 5
    # model = KNeighborsClassifier(n_neighbors=k)
    # model.fit(x_train, y_train)
    # y_pred = model.predict(x_test)
    # logging.info(f"impl_by_sklearn 准确率：{accuracy_score(y_test, y_pred)}")
    #
    # my_model = MyKNN(n_neighbors=k)
    # my_model.fit(x_train, y_train)
    # y_pred = my_model.predict(x_test)
    # logging.info(f"impl_by_ours 准确率：{accuracy_score(y_test, y_pred)}")
