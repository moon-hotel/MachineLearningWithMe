import numpy as np
import logging
import sys
from copy import deepcopy
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


class Node(object):
    def __init__(self, ):
        self.sample_index = None  # 保存当前节点中对应样本在数据集中的索引
        self.values = None  # 保存每个类别的数量 e.g. [5,10] 表示当前节点中第0个类别有5个样本，第1个类别有10个样本
        self.feature_id = -1  # 保存当前节点对应划分特征的id
        self.features = None  # 记录当前节点可用的剩余划分特征 e.g. [0,2]表示第0,2个特征在当前节点之前还没有使用过
        self.n_samples = 0  # 保存当前节点对应的样本数量
        self.left_child = None  # 保存当前节点的左孩子
        self.right_child = None  # 保存当前节点的右孩子
        self.criterion_value = 0.  # 保存当前节点对应的基尼系数
        self.split_range = None  # 预测样本划分节点时，选择左右孩子的的特征判断区间
        # e.g. [0.5,1] 如果当前样本的划分维度对应的特征取值属于区间[0.5,1]那么则进入到当前节点的左孩子中
        #              如果当前样本的划分维度对应的特征取值不属于区间[0.5,1]那么则进入到当前节点的右孩子中
        self.n_leaf = 0  # 以当前节点为根节点时其叶子节点的个数
        self.leaf_costs = 0.  # 以当前节点为根节点时其所有叶子节点的损失和

    def __str__(self):
        """
        打印节点信息
        :return:
        """
        return f"当前节点所有样本的索引({self.sample_index})\n" \
               f"当前节点的样本数量({self.n_samples})\n" \
               f"当前节点每个类别的样本数({self.values})\n" \
               f"当前节点对应的基尼指数为({round(self.criterion_value, 3)})\n" \
               f"当前节点状态时特征集中剩余特征({self.features})\n" \
               f"当前节点状态时划分特征ID({self.feature_id})\n" \
               f"当前节点状态时划分特征离散化区间为 {self.split_range}\n" \
               f"当前节点的孩子节点数量为 {self.n_leaf}\n" \
               f"当前节点的孩子节点的损失为 {self.leaf_costs}\n"


class CART(object):
    def __init__(self, min_samples_split=2,
                 epsilon=1e-5,
                 pruning=False,
                 random_state=None):
        self.root = None
        self.min_samples_split = min_samples_split  # 用来控制是否停止分裂
        self.epsilon = epsilon  # 停止标准
        self.pruning = pruning  # 是否需要进行剪枝
        self.random_state = random_state

    def _compute_gini(self, y_class):
        """
        计算基尼指数
        :param y_class:  np.array   [n,]
        :return:
        """
        y_unique = np.unique(y_class)
        if y_unique.shape[0] == 1:  # 只有一个类别
            return 0.  # 基尼指数为0
        gini = 0.
        for i in range(len(y_unique)):  # 取每个类别
            p = np.sum(y_class == y_unique[i]) / len(y_class)
            gini += p ** 2
        gini = 1 - gini
        return gini

    def _compute_gini_da(self, f_id, data):
        """
        输入特征列索引f_id以及样本数据，计算得到这一列特征中不同特征取值下的基尼指数
        :param f_id:
        :param data:
        :return: 当前特征维度下，不同特征取值时的最小基尼指数, 离散特征区间的起始索引，以及对应的样本划分索引
        e.g.
        x = np.array([[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
                      [2, 1, 1, 2, 2, 2, 0, 2, 2, 0, 0, 2, 2, 1, 1]]).transpose()
        y = np.array([1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1])
        dt = CART()
        dt.feature_values = dt._get_feature_values(x)
        dt._y = y
        # {0: [0, 0.5, 1], 1: [0, 0.5, 1], 2: [0, 0.5, 1.5, 2]}
        X = np.hstack(([x, np.arange(len(x)).reshape(-1, 1)]))
        r = dt._compute_gini_da(2, X)
        print(r)
        (0.43939393939393934, 1, array([False,  True,  True, False, False, False, False, False, False,
       False, False, False, False,  True,  True]))
        """
        feature_values = self.feature_values[f_id]  # 取当前f_id列特征对应的离散化取值情况
        logging.debug("----------")
        logging.debug(f"所有特征维度对应的离散化特征取值为 {self.feature_values}")
        logging.debug(f"当前特征维度<{f_id}>对应的离散化特征取值为 {feature_values}")
        x_feature = data[:, f_id]  # 取f_id对应的特征列
        x_ids = np.array(data[:, -1], dtype=np.int).reshape(-1)  # 样本索引
        labels = self._y[x_ids]
        logging.debug(f"当前样本对应的标签值为{labels}")
        min_gini = 99999.
        split_id = None
        split_sample_idx = None
        for i in range(len(feature_values) - 1):  # 遍历当前特征维度中，离散化特征的每个取值区间
            index = (feature_values[i] <= x_feature) & \
                    (x_feature <= feature_values[i + 1])
            # 判断特征的取值是否存在于某个离散区间中，并以此将当前节点中的样本划分为左右两个部分
            if np.sum(index) < 1.:  # 如果当前特征取值范围没有样本，则继续
                continue
            d1, y1 = data[index], labels[index]  # 根据当前特征维度的取值将样本划分为两个部分
            d2, y2 = data[~index], labels[~index]
            gini = len(y1) / len(index) * self._compute_gini(y1) + \
                   len(y2) / len(index) * self._compute_gini(y2)
            logging.debug(f"当前特征维度在不同特征取值下的基尼指数为 {gini}")
            if gini < min_gini:  # 保存当前特征维度下，能使基尼指数最小时的特征取值
                min_gini = gini
                split_id = i
                split_sample_idx = index
            if len(feature_values) == 3:
                # 只有两种特征取值时，只需要计算一次即可，e.g. 特征维度只有0,1两种取值，那么其离散化后的范围区间为[0,0.5,1]
                # 在此时只需要通过第一个区间[0,0.5]来判断计算一次即可。所以当len(feature_values) == 3时计算一次即可
                break
        logging.debug(f"当前特征维度下的最小基尼指数为 {min_gini}; "
                      f"此时对应的离散化特征取值范围为 [{feature_values[split_id]},{feature_values[split_id + 1]}]")
        return min_gini, split_id, split_sample_idx

    def _get_feature_values(self, data):
        """
        离散化每一列特征，得到取值区间。
        离散化方法为：升序排列每一列特征并去重，然后取相邻两个特征的均值作为离散化点
                   同时为了便于后续处理，还在离散化后特征的两端分别加入的特征的最大值和最小值
        :param data:
        :return:
        e.g. x = np.array([[3, 4, 5, 6, 7],
                          [2, 2, 3, 5, 8],
                          [3, 3, 8, 8, 9.]])
             _get_feature_values(x)
             {0: [2.0, 2.5, 3.0], 1: [2.0, 2.5, 3.5, 4.0], 2: [8.0, 5.5, 4.0, 5.0],
             3: [8.0, 6.5, 5.5, 6.0], 4: [8.0, 8.5, 8.0, 7.0]}
             key: 表示特征序号
             value: 表示特征离散化后的取值
        """
        n_features = data.shape[1]
        feature_values = {}
        for i in range(n_features):
            x_feature = list(set(np.sort(data[:, i])))
            tmp_values = [x_feature[0]]  # 左边插入最小值
            for j in range(1, len(x_feature)):
                tmp_values.append(round((x_feature[j - 1] + x_feature[j]) / 2, 4))
            tmp_values.append(x_feature[-1])  # 右边插入最大值
            feature_values[i] = tmp_values
        return feature_values

    def _build_tree(self, data, f_ids):
        x_ids = np.array(data[:, -1], dtype=np.int).reshape(-1)
        node = Node()
        node.sample_index = x_ids  # 当前节点所有样本的索引
        labels = self._y[x_ids]  # 当前节点所有样本对应的标签
        node.n_samples = len(labels)  # 当前节点的样本数量
        node.values = np.bincount(labels, minlength=self.n_classes)  # 当前节点每个类别的样本数
        node.features = f_ids  # 当前节点状态时特征集中剩余特征
        logging.debug("========>")
        logging.debug(f"当前节点所有样本的索引 {node.sample_index}")
        logging.debug(f"当前节点的样本数量 {node.n_samples}")
        logging.debug(f"当前节点每个类别的样本数 {node.values}")
        logging.debug(f"当前节点状态时特征集中剩余特征 {node.features}")
        if self.root is None:
            self.root = node

        y_unique = np.unique(labels)  # 当前节点中存在的类别情况
        if y_unique.shape[0] == 1 or len(f_ids) < 1 \
                or node.n_samples <= self.min_samples_split:  # 只有一个类别或特征集为空或样本数量少于min_samples_split
            return node
        gini = self._compute_gini(labels)  # 计算当前节点对应的基尼指数
        node.criterion_value = gini
        if gini < self.epsilon:
            return node
        logging.debug(f"当前节点中的样本基尼指数为 {gini}")
        min_gini = 99999
        split_id = None  # 保存所有可用划分特征中，能够值得基尼指数最小的特征 对应特征离散区间的起始索引
        split_sample_idx = None  # 最小基尼指数下对应的样本划分索引
        best_feature_id = -1  # 保存所有可用划分特征中，能够使得基尼指数最小的特征 对应的特征ID
        for f_id in f_ids:  # 遍历每个特征
            # 遍历特征下的每种取值方式的基尼指数，并返回最小的
            m_gini, s_id, s_s_idx = self._compute_gini_da(f_id, data)
            if m_gini < min_gini:  # 查找所有特征所有取值方式下，基尼指数最小的
                min_gini = m_gini
                split_id = s_id
                split_sample_idx = s_s_idx
                best_feature_id = f_id
        node.feature_id = best_feature_id
        feature_values = self.feature_values[best_feature_id]
        node.split_range = [feature_values[split_id], feature_values[split_id + 1]]
        logging.debug(f"【***此时选择第{best_feature_id}个特征进行样本划分，"
                      f"此时第{best_feature_id}个特征对应的离散化特征取值范围为 {node.split_range}，"
                      f"最小基尼指数为 {min_gini}***】")
        left_data = data[split_sample_idx]
        right_data = data[~split_sample_idx]
        candidate_ids = deepcopy(f_ids)
        candidate_ids.remove(best_feature_id)  # 当前节点划分后的剩余特征集
        if len(left_data) > 0:
            node.left_child = self._build_tree(left_data, candidate_ids)  # 递归构建决策树
        if len(right_data) > 0:
            node.right_child = self._build_tree(right_data, candidate_ids)
        return node

    def fit(self, X, y):
        """
        :param X: shape: [n_samples, n_features]
        :param y: shape: [n_samples,]
        :return:
        """
        if self.pruning:  # 如果剪枝则划分一部分数据作为测试集
            X, self.x_test, y, self.y_test = train_test_split(X, y,
                                                              test_size=0.1,
                                                              random_state=self.random_state)
        self._y = np.array(y).reshape(-1)
        self.n_classes = len(np.bincount(y))  # 得到当前数据集的类别数量
        feature_ids = [i for i in range(X.shape[1])]  # 得到特征的序号
        self.feature_values = self._get_feature_values(X)  # 得到离散化特征
        self._X = np.hstack(([X, np.arange(len(X)).reshape(-1, 1)]))
        # 将训练集中每个样本的序号加入到X的最后一列
        self._build_tree(self._X, feature_ids)  # 递归构建决策树
        if self.pruning:  # 进行剪枝
            self._pruning_leaf()

    def level_order(self, return_node=False):
        """
        层次遍历
        :return:
        """
        logging.debug("===============================")
        logging.debug("正在进行层次遍历……")
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
        if return_node:
            logging.debug("并返回层次遍历的所有结果！")
            return res  # 按层次遍历的顺序返回各层节点的地址
            # [[root], [level2 node1, level2_node2], [level3,...] [level4,...],...[],]
        logging.debug("层次遍历结果为：")
        for i, r in enumerate(res):
            logging.debug(f"第{i + 1}层的节点为：")
            for node in r:
                logging.debug(node)
            logging.debug("\n")

    def _predict_one_sample(self, x):
        """
        预测单一样本
        :param x: [n_features,]
        :return:
        """
        current_node = self.root
        while True:
            # 有些情况下叶子节点没有兄弟节点
            if not current_node.left_child or \
                    not current_node.right_child or \
                    current_node.split_range is None:
                # 当前节点为叶子节点
                return current_node.values
            current_feature_id = current_node.feature_id
            current_feature = x[current_feature_id]
            split_range = current_node.split_range
            if split_range[0] <= current_feature <= split_range[1]:
                current_node = current_node.left_child
            else:
                current_node = current_node.right_child

    def predict(self, X):
        """
        :param X: shape [n_samples,n_features]
        :return:
        """
        results = []
        for x in X:
            results.append(self._predict_one_sample(x))
        results = np.array(results)
        logging.debug(f"原始预测结果为:\n{results}")
        y_pred = np.argmax(results, axis=1)
        return y_pred

    def _get_pruning_gt(self, node):
        """
        计算对当前节点剪枝前和剪枝后对应的gt值
        :return:
        """

        def _compute_cost_in_leaf(labels):
            """
            计算节点的损失  c = -\sum_{k=1}^KN_{tk}\log{\frac{N_{tk}}{N_t}}
            :param labels:
            :return:
            e.g. y_labels = np.array([1, 1, 1, 0])
            _compute_cost_in_leaf(y_labels)   3.24511
            """
            y_count = np.bincount(labels)
            n_samples = len(labels)
            cost = 0
            for i in range(len(y_count)):
                if y_count[i] == 0:
                    continue
                cost += y_count[i] * np.log2(y_count[i] / n_samples)
            return -cost

        if not node.left_child and not node.right_child:
            node.leaf_costs = _compute_cost_in_leaf(self._y[node.sample_index])
            # 如果当前节点是叶子节点，则计算该叶子节点对应的损失值
            return 99999.
        parent_cost = _compute_cost_in_leaf(self._y[node.sample_index])  # 计算以当前节点为根节点剪枝后的损失
        if node.left_child:
            node.leaf_costs += node.left_child.leaf_costs  # 以当前节点为根节点累计剪枝前所有叶子节点的损失
        if node.right_child:
            node.leaf_costs += node.right_child.leaf_costs
        g_t = (parent_cost - node.leaf_costs) / (node.n_leaf - 1 + 1e-5)  # 计算gt，其中1e-5为平滑项
        logging.debug(f"------------------")
        logging.debug(f"当前节点gt为:{g_t}")
        logging.debug(f"当前节点（剪枝后）的损失为：{parent_cost}")
        logging.debug(f"当前节点的孩子节点（剪枝前）损失为：{node.leaf_costs}")
        return g_t

    def _get_subtree_sequence(self):
        subtrees = []
        logging.debug("\n\n")
        logging.debug(f"正在获取子序列T0,T1,T2,T3...")
        stop = False
        while not stop:
            if not self.root.right_child and not self.root.left_child:
                stop = True
            # while self.root.right_child and self.root.left_child:
            level_order_nodes = self.level_order(return_node=True)
            best_gt = 99999.
            best_pruning_node = None
            for i in range(len(level_order_nodes) - 1, -1, -1):  # 从对底层向上遍历
                current_level_nodes = level_order_nodes[i]  # 取第i层的所有节点
                for j in range(len(current_level_nodes)):  # 从左向右遍历
                    current_node = current_level_nodes[j]  # 取第i层的第j个节点
                    current_node.n_leaf = 0  # 对于每一颗子树来说，重置计数，因为原始值中包含有上一课子树的计数信息
                    current_node.leaf_costs = 0.  # 因为需要在每一颗子树中保存相关信息
                    if current_node.left_child is not None:
                        current_node.n_leaf += current_node.left_child.n_leaf  # 计算以当前节点为根节点的叶子节点数
                    if current_node.right_child is not None:
                        current_node.n_leaf += current_node.right_child.n_leaf
                    elif not current_node.left_child and not current_node.right_child:
                        current_node.n_leaf = 1  # 当前节点为叶子节点，则其对应的叶子节点数为1
                    gt = self._get_pruning_gt(current_node)
                    if gt < best_gt:
                        best_gt = gt
                        best_pruning_node = current_node
            logging.debug(f"本轮结束，最小的gt为 {best_gt} #######")
            subtrees.append(deepcopy(self.root))
            if not stop:
                best_pruning_node.left_child = None
                best_pruning_node.right_child = None  # 剪枝
        return subtrees

    def _pruning_leaf(self):
        subtrees = self._get_subtree_sequence()  # 得到所有子树序列T0,T1,T2,...,Tn
        best_tree = None
        max_acc = 0.
        for tree in subtrees:  # 在测试集上对所有子树进行测试，
            self.root = tree  # 选择准确率最高的子树作为最终的决策树
            acc = accuracy_score(self.predict(self.x_test), self.y_test)
            if acc > max_acc:
                max_acc = acc
                best_tree = tree
        self.root = best_tree


def load_simple_data():
    x = np.array([[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
                  [2, 1, 1, 2, 2, 2, 0, 2, 2, 0, 0, 2, 2, 1, 1]]).transpose()
    y = np.array([1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1])
    return x, y


def test_gini():
    x, y = load_simple_data()
    dt = CART()
    logging.info(f"标签{y}的GINI指数为: {dt._compute_gini(y)}")


def test_cart():
    x, y = load_simple_data()
    dt = CART(min_samples_split=1)
    dt.fit(x, y)
    dt.level_order()
    y_pred = dt.predict(np.array([[0, 0, 2],
                                  [0, 1, 1],
                                  [1, 1, 1],
                                  [0, 1, 0],
                                  [0, 1, 2]]))
    logging.info(f"CART 预测结果为：{y_pred}")


def test_wine_classification():
    x, y = load_wine(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2022)
    dt = CART(min_samples_split=2, pruning=True, random_state=16)
    dt.fit(x_train, y_train)
    y_pred = dt.predict(x_test)
    logging.info(f"CART 准确率：{accuracy_score(y_test, y_pred)}")

    model = DecisionTreeClassifier(criterion='gini')
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    logging.info(f"DecisionTreeClassifier 准确率：{accuracy_score(y_test, y_pred)}")


def count_children():
    x, y = load_simple_data()
    dt = CART(min_samples_split=2)
    dt.fit(x, y)
    level_order_nodes = dt.level_order(return_node=True)
    for i in range(len(level_order_nodes) - 1, -1, -1):
        current_level_nodes = level_order_nodes[i]  # 取第i层的所有节点
        for j in range(len(current_level_nodes)):
            current_node = current_level_nodes[j]  # 取第i层的第j个节点
            if current_node.left_child is not None:
                current_node.n_leaf += (current_node.left_child.n_leaf)
            if current_node.right_child is not None:
                current_node.n_leaf += (current_node.right_child.n_leaf)
            if not current_node.left_child and not current_node.right_child:
                current_node.n_leaf = 1

    dt.level_order()


def test_get_subtree():
    x, y = load_simple_data()
    dt = CART(min_samples_split=1)
    dt.fit(x, y)
    subtrees = dt._get_subtree_sequence()
    logging.debug(f"生成子树个数为：{len(subtrees)}")
    for i, tree in enumerate(subtrees):
        logging.debug(f"-----正在层次遍历第 {i} 颗子树-----")
        dt.root = tree
        dt.level_order()


if __name__ == '__main__':
    formatter = '[%(asctime)s] - %(levelname)s: %(message)s'
    logging.basicConfig(level=logging.INFO,  # 如果需要查看详细信息可将该参数改为logging.DEBUG
                        format=formatter,  # 关于Logging模块的详细使用可参加文章https://www.ylkz.life/tools/p10958151/
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[logging.StreamHandler(sys.stdout)])
    test_gini()

    test_cart()
    # - DEBUG: ========>
    # - DEBUG: 当前节点所有样本的索引 [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14]
    # - DEBUG: 当前节点的样本数量 15
    # - DEBUG: 当前节点每个类别的样本数 [ 5 10]
    # - DEBUG: 当前节点状态时特征集中剩余特征 [0, 1, 2]
    # - DEBUG: 当前节点中的样本基尼指数为 0.4444444444444444
    # - DEBUG: ----------
    # - DEBUG: 所有特征维度对应的离散化特征取值为 {0: [0, 0.5, 1], 1: [0, 0.5, 1], 2: [0, 0.5, 1.5, 2]}
    # - DEBUG: 当前特征维度<0>对应的离散化特征取值为 [0, 0.5, 1]
    # - DEBUG: 当前样本对应的标签值为[1 1 1 0 0 0 0 1 1 1 1 1 1 0 1]
    # - DEBUG: 当前特征维度在不同特征取值下的基尼指数为 0.3452380952380953
    # - DEBUG: 当前特征维度下的最小基尼指数为 0.3452380952380953; 此时对应的离散化特征取值范围为 [0,0.5]
    # - DEBUG: ----------
    # - DEBUG: 所有特征维度对应的离散化特征取值为 {0: [0, 0.5, 1], 1: [0, 0.5, 1], 2: [0, 0.5, 1.5, 2]}
    # - DEBUG: 当前特征维度<1>对应的离散化特征取值为 [0, 0.5, 1]
    # - DEBUG: 当前样本对应的标签值为[1 1 1 0 0 0 0 1 1 1 1 1 1 0 1]
    # - DEBUG: 当前特征维度在不同特征取值下的基尼指数为 0.380952380952381
    # - DEBUG: 当前特征维度下的最小基尼指数为 0.380952380952381; 此时对应的离散化特征取值范围为 [0,0.5]
    # - DEBUG: ----------
    # - DEBUG: 所有特征维度对应的离散化特征取值为 {0: [0, 0.5, 1], 1: [0, 0.5, 1], 2: [0, 0.5, 1.5, 2]}
    # - DEBUG: 当前特征维度<2>对应的离散化特征取值为 [0, 0.5, 1.5, 2]
    # - DEBUG: 当前样本对应的标签值为[1 1 1 0 0 0 0 1 1 1 1 1 1 0 1]
    # - DEBUG: 当前特征维度在不同特征取值下的基尼指数为 0.4444444444444445
    # - DEBUG: 当前特征维度在不同特征取值下的基尼指数为 0.43939393939393934
    # - DEBUG: 当前特征维度在不同特征取值下的基尼指数为 0.44047619047619047
    # - DEBUG: 当前特征维度下的最小基尼指数为 0.43939393939393934; 此时对应的离散化特征取值范围为 [0.5,1.5]
    # - DEBUG: 【***此时选择第0个特征进行样本划分，此时第0个特征对应的离散化特征取值范围为 [0, 0.5]，最小基尼指数为 0.3452380952380953***】
    # ................
    #  DEBUG: 正在进行层次遍历……
    # - DEBUG: 层次遍历结果为：
    # - DEBUG: 第1层的节点为：
    # - DEBUG: 当前节点所有样本的索引([ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14])
    # 当前节点的样本数量(15)
    # 当前节点每个类别的样本数([ 5 10])
    # 当前节点对应的基尼指数为(0.444)
    # 当前节点状态时特征集中剩余特征([0, 1, 2])
    # 当前节点状态时划分特征ID(0)
    # 当前节点状态时划分特征离散化区间为 [0, 0.5]
    # 当前节点的孩子节点数量为 0
    # 当前节点的孩子节点的损失为 0.0
    #
    # - DEBUG:
    #
    # - DEBUG: 第2层的节点为：
    # - DEBUG: 当前节点所有样本的索引([0 1 2 3 4 5 6])
    # 当前节点的样本数量(7)
    # 当前节点每个类别的样本数([4 3])
    # 当前节点对应的基尼指数为(0.49)
    # 当前节点状态时特征集中剩余特征([1, 2])
    # 当前节点状态时划分特征ID(1)
    # 当前节点状态时划分特征离散化区间为 [0, 0.5]
    # 当前节点的孩子节点数量为 0
    # 当前节点的孩子节点的损失为 0.0

    # test_get_subtree()
    # test_wine_classification()
