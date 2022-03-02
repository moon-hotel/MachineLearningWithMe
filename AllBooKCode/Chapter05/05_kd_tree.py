import numpy as np
from sklearn.neighbors import KDTree


class Node(object):
    """
    定义KD树中的节点信息
    """

    def __init__(self, data=None):
        self.data = data
        self.left_child = None
        self.right_child = None

    def __str__(self):
        """
        打印节点信息
        :return:
        """
        return f"data({self.data})"


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
        dim: 样本点的维度
    """

    def __init__(self, points):
        self.root = None
        self.dim = points.shape[1]
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
        print("当前待划分样本点：", data)
        idx = len(data) // 2
        node = Node(data[idx])
        print("父节点：", data[idx])
        left_data = data[:idx]
        print("左子树: ", left_data)
        right_data = data[idx + 1:]
        print("右子树: ", right_data)
        print("============")
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
                tmp.append(node.data)
                if node.left_child:
                    queue.append(node.left_child)
                if node.right_child:
                    queue.append(node.right_child)
            res.append(tmp)
        print("\n层次遍历结果为：")
        for i, r in enumerate(res):
            print(f"第{i + 1}层的节点为：{r}")

    def nearest_search(self, point):
        """
        最近邻搜索
        :param point: 必须为np.array()类型，形状为 [n,]
        :return:
        """
        best_node = None
        best_dist = np.inf
        visited = []  # 用来记录哪些节点被访问过

        def nearest_node_search(point, curr_node, order=0):
            """
            :param point:  shape: (n,)
            :param curr_node: data: shape (n,)
            :param order:
            :return:
            """
            nonlocal best_node, best_dist, visited  # 声明这三个变量不是局部变量
            print(f"当前访问节点为：{curr_node}")
            visited.append(curr_node)
            if curr_node is None:
                return None
            dist = distance(curr_node.data, point)
            print(f"当前访问节点到被搜索点的距离为：{dist},全局最佳距离为：{best_dist}, 全局最佳点为：{best_node}\n")
            if dist < best_dist:
                best_dist = dist
                best_node = curr_node
            cmp_dim = order % self.dim
            if point[cmp_dim] < curr_node.data[cmp_dim]:
                nearest_node_search(point, curr_node.left_child, order + 1)
            else:
                nearest_node_search(point, curr_node.right_child, order + 1)
            if np.abs(curr_node.data[cmp_dim] - point[cmp_dim]) < best_dist:
                child = curr_node.left_child if curr_node.left_child not in visited else curr_node.right_child
                nearest_node_search(point, child, order + 1)

        nearest_node_search(point, self.root, 0)
        return best_node, best_dist

    @staticmethod
    def append(k_nearest_nodes, curr_node, point):
        """
        将当前节点加入到k_nearest_nodes中并按每个节点到被搜索点的距离升序排列
        :param k_nearest_nodes: list，用来保存到目前位置距离被搜索点最近的K个点
        :param curr_node:  Node()类型，当前访问的节点
        :param point: 被搜索点，np.array()类型，形状为(m,)
        :return:
        """
        k_nearest_nodes.append(curr_node)
        k_nearest_nodes = sorted(k_nearest_nodes,
                                 key=lambda x: distance(x.data, point))
        print(f"        当前K近邻中的节点为（已按距离排序）：", end='')
        for item in k_nearest_nodes:
            print(item, end='\t')
        print()
        return k_nearest_nodes

    def k_nearest_search(self, points, k):
        """
        分别查找points中每个样本距其最近的k个样本点
        :param points: np.array()   形状为[n,m]
        :param k:
        :return:
        """
        all_results = []
        for point in points:
            k_nodes = self._k_nearest_search(point, k)
            all_results.append([node.data for node in k_nodes])
        return np.array(all_results)

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
        print(f"\n\n正在查找离样本点{point}最近的{k}个样本点！")

        def k_nearest_node_search(point, curr_node, order=0):
            nonlocal k_nearest_nodes, n
            print(f"    K近邻搜索当前访问节点为：{curr_node}")
            if curr_node is None:
                return None
            visited.append(curr_node)
            if n < k: # 如果当前还没找到k个点，则直接进行保存
                n += 1
                k_nearest_nodes = self.append(k_nearest_nodes, curr_node, point)
            else: # 已经找到k个局部最优点，开始进行筛选
                dist = (distance(curr_node.data, point) < distance(point, k_nearest_nodes[-1].data))
                if dist:
                    k_nearest_nodes.pop()  # 移除最后一个
                    k_nearest_nodes = self.append(k_nearest_nodes, curr_node, point) # 加入新的点并进行排序
            cmp_dim = order % self.dim
            if point[cmp_dim] < curr_node.data[cmp_dim]:
                k_nearest_node_search(point, curr_node.left_child, order + 1)
            else:
                k_nearest_node_search(point, curr_node.right_child, order + 1)
            if n < k or np.abs(curr_node.data[cmp_dim] - point[cmp_dim]) < distance(point,
                                                                                    k_nearest_nodes[-1].data):
                child = curr_node.left_child if curr_node.left_child not in visited else curr_node.right_child
                k_nearest_node_search(point, child, order + 1)

        k_nearest_node_search(point, self.root, 0)
        return k_nearest_nodes

def test_kd_tree_build_and_search():
    points = np.array(
        [[5, 7], [3, 8], [6, 3], [8, 5], [15, 6.], [10, 4], [12, 13], [9, 10], [11, 14]])
    print("MyKDTree 运行结果：")
    print("构建KD树")
    tree = MyKDTree(points)
    print("构建结束")
    print("MyKDTree 层次遍历结果：")
    tree.level_order()

    best_node, best_dist = tree.nearest_search(np.array([8.9, 4]))
    print(best_node)
    print(best_dist)
    k_best_nodes = tree.k_nearest_search(np.array([[10, 3], [8.9, 4], [2, 9.], [5, 5]]), k=3)
    print(k_best_nodes)

    print("sklearn KDTree 运行结果：")
    kd_tree = KDTree(points)
    dist, ind = kd_tree.query(np.array([[10, 3], [8.9, 4], [2, 9.], [5, 5]]), k=3)
    print(points[ind])


if __name__ == '__main__':
    test_kd_tree_build_and_search()
