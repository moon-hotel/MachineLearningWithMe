import numpy as np
import logging
import sys
import copy

from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


class Node(object):
    def __init__(self, ):
        self.sample_index = None  # 保存当前节点中对应样本在数据集中的索引
        self.values = None  # 保存每个类别的数量
        self.features = None  # 保存当前节点状态时特征集中剩余特征
        self.feature_id = -1  # 保存当前节点对应划分特征的id
        self.label = None  # 保存当前节点对应的类别标签（叶子节点才有）
        self.n_samples = 0  # 保存当前节点对应的样本数量
        self.children = {}  # 保存当前节点对应的孩子节点
        self.criterion_value = 0.
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
               f"当前节点对应的信息增益（比）({round(self.criterion_value, 3)})\n" \
               f"当前节点状态时特征集中剩余特征({self.features})\n" \
               f"当前节点状态时划分特征ID({self.feature_id})\n" \
               f"当前节点对应的类别标签为({self.label})\n" \
               f"当前节点为根节点对应孩子节点数为({self.n_leaf})\n" \
               f"当前节点为根节点对应孩子节点损失为({self.leaf_costs})\n" \
               f"当前节点对应的孩子为({self.children.keys()})"


class DecisionTree(object):
    def __init__(self, min_samples_split=2,
                 criterion='id3',
                 epsilon=1e-5,
                 alpha=0.):
        self.root = None
        self.min_samples_split = min_samples_split  # 用来控制是否停止分裂
        self.epsilon = epsilon  # 用来控制是否停止分裂
        self.criterion = criterion  # 划分标注，ID3还是C4.5
        self.alpha = alpha
        # criterion = "id3" 表示使用ID3进行决策树构建
        # criterion = "c45" 表示使用C4.5进行决策树

    def _compute_entropy(self, y_class):
        """
        计算信息熵（主要用于ID3中计算样本的信息熵，以及C4.5中特征的信息熵）
        :param y_class:  np.array   [n,]
        :return:
        """
        y_unique = np.unique(y_class)
        if y_unique.shape[0] == 1:  # 只有一个类别
            return 0.  # 熵为0
        ety = 0.
        for i in range(len(y_unique)):  # 取每个类别
            p = np.sum(np.abs(y_class - y_unique[i]) < 0.0001) / len(y_class)
            # 因为同时要计算特征信息熵，而特征可能是浮点型所以不能用==来进行判断
            ety += p * np.log2(p)
        return -ety

    def _compute_condition_entropy(self, X_feature, id, y_class):
        """
        计算条件熵
        :param X_feature: shape: [n,]
        :param y_class: shape: [n,]
        :return:
        """
        f_unique = self.feature_values[id]  # 取当前id列特征对应的取值情况
        logging.debug(f"当前特征维度的离散取值情况为：{f_unique}")
        result = 0.
        for i in range(len(f_unique) - 1):  # 取每个特征类别
            index_x = (f_unique[i] <= X_feature) & (X_feature <= f_unique[i + 1])
            # 离散化后的特征则判断当前特征值是否存在于某个区间
            p = np.sum(index_x) / len(y_class)  # 取索引对应的标签
            ety = self._compute_entropy(y_class[index_x])  # 计算标签对应的信息熵
            result += p * ety  # 计算条件熵
        return result

    def fit(self, X, y):
        """
        :param X: shape: [n_samples, n_features]
        :param y: shape: [n_samples,]
        :return:
        """
        self._y = np.array(y).reshape(-1)
        self.n_classes = len(np.bincount(y))  # 得到当前数据集的类别数量
        feature_ids = [i for i in range(X.shape[1])]  # 得到特征的序号
        self.feature_values = self._get_feature_values(X)
        self._X = np.hstack(([X, np.arange(len(X)).reshape(-1, 1)]))
        # 将训练集中每个样本的序号加入到X的最后一列
        self._build_tree(self._X, feature_ids)  # 递归构建决策树
        self._pruning_leaf()

    @staticmethod
    def _get_label(labels):
        """
        根据给定标签，返回出现次数最多的类别

        :param labels: ['1', '1', '1', '0', '0', '0', '0', '2']
        :return: '0'
        """
        r = {}
        for i in range(len(labels)):
            r[labels[i]] = r.setdefault(labels[i], 0) + 1
        return sorted(r.items(), key=lambda x: x[1])[-1][0]

    def _split_criterion(self, ety, X_feature, id, y_class):
        c_ety = self._compute_condition_entropy(X_feature, id, y_class)  # 计算每个特征下的条件熵
        logging.debug(f"当前节点下特征对应的条件熵为 {c_ety}")
        info_gains = ety - c_ety  # 计算信息增益
        if self.criterion == "id3":
            return info_gains
        elif self.criterion == "c45":
            f_ety = self._compute_entropy(X_feature)
            logging.debug(f"当前节点下特征对应的信息熵为 {f_ety}")
            return info_gains / f_ety
        else:
            raise ValueError(f"划分标准 self.criterion = {self.criterion}只能是 id3 和 c45 其中之一！")

    def _build_tree(self, data, f_ids):
        """
        :param x_ids: np.array() [n,] 样本索引，用于在节点中保存每个样本的索引，以及根据索引取到对应样本
        :param f_ids: list 特征序号，用于在当前节点中保存特征集中还剩余哪些特征
        :return:
        """
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

        y_unique = np.unique(labels)  # 当前节点一种存在的类别数量
        if y_unique.shape[0] == 1 or len(f_ids) < 1 \
                or node.n_samples <= self.min_samples_split:  # 只有一个类别或特征集为空或样本数量少于3
            node.label = self._get_label(labels)  # 根据多数原则确定当前节点对应的类别
            return node
        ety = self._compute_entropy(labels)  # 计算当前节点所有样本对应的信息熵
        node.criterion_value = ety
        logging.debug(f"当前节点中的样本信息熵为 {ety}")
        max_criterion = 0
        best_feature_id = -1
        for id in f_ids:  # 根据划分标准（如信息增益）选择最佳特征
            criterion = self._split_criterion(ety, data[:, id], id, labels)
            logging.debug(f"当前节点第{id}个特征在标准{self.criterion}下对应的划分指标为{criterion}")
            if criterion > max_criterion:  # 遍历选择最大指标（信息增益或信息增益比）
                max_criterion = criterion
                best_feature_id = id
        if max_criterion < self.epsilon:  # 最大指标小于设定阈值
            node.label = self._get_label(labels)  # 根据多数原则确定当前节点对应的类别
            return node
        node.feature_id = best_feature_id
        logging.debug(f"此时选择第{best_feature_id}个特征进行样本划分")
        feature_values = self.feature_values[best_feature_id]  # 得到当前特征离散化后特征的取值情况
        logging.debug(f"此时第{best_feature_id}个特征的取值为{feature_values}")
        candidate_ids = copy.copy(f_ids)
        candidate_ids.remove(best_feature_id)  # 当前节点划分后的剩余特征集

        for i in range(len(feature_values) - 1):  # 依次遍历特征离散化后的每个取值情况
            logging.debug(f"正在遍历最佳特征（第{best_feature_id}个）的取值 {feature_values[i + 1]}")
            v = data[:, best_feature_id]
            c = (feature_values[i] <= v) & (v <= feature_values[i + 1])  # 根据当前特征维度的取值，来取对应的特征维度
            index = np.array([i for i in range(len(c)) if c[i] == True])  # 获取对应的索引
            if len(index) == 0:  # 如果当前特征取值范围没有样本，则继续
                continue
            node.children[str(feature_values[i + 1])] = self._build_tree(data[index], candidate_ids)
            # 由于离散化的特征值有浮点型，所以这里转换成字符串来进行保存
        return node

    def level_order(self, return_node=False):
        """
        层次遍历
        :return:
        """
        logging.debug("\n\n正在进行层次遍历……")
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
                for k, v in node.children.items():
                    queue.append(v)
            res.append(tmp)
        if return_node:
            return res  # 按层次遍历的顺序返回各层节点的地址
            # [[root], [level2 node1, level2_node2], [level3,...] [level4,...],...[],]
        logging.debug("\n ======层次遍历结果为=======")
        for i, r in enumerate(res):
            logging.debug(f"<==========第{i + 1}层的节点为==========>")
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
            if len(current_node.children) < 1:
                # ① 当前节点为叶子节点
                return current_node.values
            current_feature_id = current_node.feature_id
            current_feature = x[current_feature_id]
            current_feature_values = self.feature_values[current_feature_id]
            exist_child = False
            for i in range(len(current_feature_values) - 1):
                if current_feature_values[i] <= current_feature <= current_feature_values[i + 1]:
                    exist_child = True
                    if str(current_feature_values[i + 1]) not in current_node.children:
                        # 由于数据集不充分当前节点的孩子节点不存在下一个划分节点的某一个取值
                        # 例如根据测试数据集load_simple_data（）构造得到的id3树，对于特征[0,1,0]来说，
                        # 在遍历最后一个特征维度时，取值0就不存在于孩子节点中
                        return current_node.values
                    current_node = current_node.children[str(current_feature_values[i + 1])]
            if not exist_child:
                return current_node.values

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

    def _pruning_leaf(self):
        level_order_nodes = self.level_order(return_node=True)
        # 获取得到层次遍历的所有结果
        logging.debug(f"正在进行剪枝操作……")
        for i in range(len(level_order_nodes) - 1, -1, -1):
            # 从下往上依次遍历每一层节点
            current_level_nodes = level_order_nodes[i]  # 取第i层的所有节点
            for j in range(len(current_level_nodes)):
                current_node = current_level_nodes[j]  # 取第i层的第j个节点
                if len(current_node.children) == 0:  # 当前节点为叶子节点时
                    current_node.n_leaf = 1  # 令其叶节点个数为1
                else:
                    for _, leaf_node in current_node.children.items():
                        current_node.n_leaf += leaf_node.n_leaf  # 统计以当前节点为根节点时的叶子节点数量
                if self._is_pruning_leaf(current_node):
                    current_node.children = {}
                    current_node.n_leaf = 1

    def _is_pruning_leaf(self, node):
        """
        判断是否对当前节点进行剪枝
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

        if len(node.children) < 1:  # 当前节点为叶子节点时，计算叶子节点对应的损失
            node.leaf_costs = _compute_cost_in_leaf(self._y[node.sample_index])
            return False
        parent_cost = _compute_cost_in_leaf(self._y[node.sample_index])  # 剪枝后的损失
        for (_, leaf_node) in node.children.items():  # 剪枝前累加所有叶子节点的损失
            node.leaf_costs += leaf_node.leaf_costs
        logging.debug(f"当前节点的损失为：{parent_cost} + {self.alpha}")
        logging.debug(f"当前节点的孩子节点损失和为：{node.leaf_costs} + {self.alpha} * {node.n_leaf}")
        if node.leaf_costs + self.alpha * node.n_leaf > parent_cost + self.alpha:
            #  当剪枝前的损失  >  剪枝后的损失， 则表示当前节点可以进行剪枝（减掉其所有孩子）
            return True
        return False

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
             {0: [2.0, 2.5, 3.0], 1: [2.0, 2.5, 3.5, 4.0], 2: [3.0, 4.0, 6.5, 8.0],
             3: [5.0, 5.5, 7.0, 8.0], 4: [7.0, 7.5, 8.5, 9.0]}
             key: 表示特征序号
             value: 表示特征离散化后的取值
        """
        n_features = data.shape[1]
        feature_values = {}
        for i in range(n_features):
            x_feature = sorted(set(data[:, i]))  # 去重与排序
            tmp_values = [x_feature[0]]  # 左边插入最小值
            for j in range(1, len(x_feature)):
                tmp_values.append(round((x_feature[j - 1] + x_feature[j]) / 2, 4))
            tmp_values.append(x_feature[-1])  # 右边插入最大值
            feature_values[i] = tmp_values
        return feature_values


def test_decision_tree_pruning():
    x, y = load_simple_data()
    dt = DecisionTree(criterion='id3', alpha=0.)
    dt.fit(x, y)
    x_test = np.array([[0, 1, 1],
                       [0, 1, 2]])
    logging.debug(f"剪枝前的层次遍历结果")
    # dt.level_order()
    logging.info(f"剪枝前的预测类别：{dt.predict(x_test)}")

    dt = DecisionTree(criterion='id3', alpha=1.25)
    dt.fit(x, y)
    logging.debug(f"剪枝后的层次遍历结果")
    # dt.level_order()
    logging.info(f"剪枝后的预测类别：{dt.predict(x_test)}")


def load_simple_data():
    x = np.array([[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
                  [2, 1, 1, 2, 2, 2, 0, 2, 2, 0, 0, 2, 2, 1, 1]]).transpose()
    y = np.array([1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1])
    return x, y


def test_decision_tree():
    x, y = load_simple_data()
    dt = DecisionTree(criterion='c45')
    dt.fit(x, y)
    dt.level_order()
    y_pred = dt.predict(np.array([[0, 0, 2],
                                  [0, 1, 1],
                                  [0, 1, 0],
                                  [0, 1, 2]]))
    logging.info(f"DecisionTree 预测结果为：{y_pred}")


def test_iris_classification():
    x, y = load_iris(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2022)
    dt = DecisionTree(criterion='id3', alpha=1.6)
    dt.fit(x_train, y_train)
    y_pred = dt.predict(x_test)
    logging.info(f"DecisionTree 准确率：{accuracy_score(y_test, y_pred)}")

    model = DecisionTreeClassifier(criterion='entropy')
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    logging.info(f"DecisionTreeClassifier 准确率：{accuracy_score(y_test, y_pred)}")


if __name__ == '__main__':
    formatter = '[%(asctime)s] - %(levelname)s: %(message)s'
    logging.basicConfig(level=logging.INFO,  # 如果需要查看简略信息可将该参数改为logging.INFO
                        format=formatter,  # 关于Logging模块的详细使用可参加文章https://www.ylkz.life/tools/p10958151/
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[logging.StreamHandler(sys.stdout)])
    # test_decision_tree()
    test_iris_classification()
    # test_decision_tree_pruning()
