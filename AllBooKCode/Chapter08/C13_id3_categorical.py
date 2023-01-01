import numpy as np
import logging
import sys
import copy
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

sys.path.append('../')
from utils import load_cut_spam
from utils import VectWithoutFrequency


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
        return f"<======================>\n" \
               f"当前节点所有样本的索引({self.sample_index})\n" \
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
            p = np.sum(y_class == y_unique[i]) / len(y_class)
            ety += p * np.log2(p)
        return -ety

    def _compute_condition_entropy(self, X_feature, y_class):
        """
        计算条件熵
        :param X_feature: shape: [n,]
        :param y_class: shape: [n,]
        :return:
        """
        f_unique = np.unique(X_feature)
        result = 0.
        for i in range(len(f_unique)):  # 取每个特征类别
            index_x = (X_feature == f_unique[i])  # 取当前特征类别对应的索引
            p = np.sum(index_x) / len(y_class)  # 取索引对应的标签
            ety = self._compute_entropy(y_class[index_x])  # 计算标签对应的信息熵
            result += p * ety  # 计算条件熵
        return result

    def fit(self, X, y):
        """
        输入的数据集X特征必须为categorical类型
        :param X: shape: [n_samples, n_features]
        :param y: shape: [n_samples,]
        :return:
        """
        self._y = np.array(y).reshape(-1)
        self.n_classes = len(np.bincount(y))  # 得到当前数据集的类别数量
        feature_ids = [i for i in range(X.shape[1])]  # 得到特征的序号
        self._X = np.hstack(([X, np.arange(len(X)).reshape(-1, 1)]))
        # 将训练集中每个样本的序号加入到X的最后一列
        self._build_tree(self._X, feature_ids)  # 递归构建决策树
        self._pruning_leaf()
        return self

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

    def _split_criterion(self, ety, X_feature, y_class):
        c_ety = self._compute_condition_entropy(X_feature, y_class)  # 计算每个特征下的条件熵
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
                or node.n_samples <= self.min_samples_split:  # 只有一个类别或特征集为空或样本数量少于min_samples_split
            node.label = self._get_label(labels)  # 根据多数原则确定当前节点对应的类别
            return node
        ety = self._compute_entropy(labels)  # 计算当前节点所有样本对应的信息熵
        node.criterion_value = ety
        logging.debug(f"当前节点中的样本信息熵为 {ety}")
        max_criterion = 0
        best_feature_id = -1
        for id in f_ids:  # 根据划分标准（如信息增益）选择最佳特征
            criterion = self._split_criterion(ety, data[:, id], labels)
            logging.debug(f"当前节点第{id}个特征在标准{self.criterion}下对应的划分指标为{criterion}")
            if criterion > max_criterion:  # 遍历选择最大指标（信息增益或信息增益比）
                max_criterion = criterion
                best_feature_id = id
        if max_criterion < self.epsilon:  # 最大指标小于设定阈值
            node.label = self._get_label(labels)  # 根据多数原则确定当前节点对应的类别
            return node
        node.feature_id = best_feature_id
        logging.debug(f"此时选择第{best_feature_id}个特征进行样本划分")
        feature_values = np.unique(data[:, best_feature_id])  # 划分特征的取值情况
        logging.debug(f"此时第{best_feature_id}个特征的取值为{feature_values}")
        candidate_ids = copy.copy(f_ids)
        candidate_ids.remove(best_feature_id)  # 当前节点划分后的剩余特征集

        for f in feature_values:  # 依次遍历每个取值情况
            logging.debug(f"正在遍历最佳特征（第{best_feature_id}个）的取值 {f}")
            c = data[:, best_feature_id] == f  # 根据当前特征维度的取值，来判断对应的样本
            index = np.array([i for i in range(len(c)) if c[i] == True])  # 获取对应的索引
            node.children[f] = self._build_tree(data[index], candidate_ids)
        return node

    def level_order(self, return_node=False):
        """
        层次遍历
        :return:
        """
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
                for k, v in node.children.items():
                    queue.append(v)
            res.append(tmp)
        if return_node:
            logging.debug(f"并返回层次遍历后的结果\n")
            return res  # 按层次遍历的顺序返回各层节点的地址
            # [[root], [level2 node1, level2_node2], [level3,...] [level4,...],...[],]
        logging.debug("\n层次遍历结果为：")
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
            current_feature_id = current_node.feature_id
            current_feature = x[current_feature_id]
            if len(current_node.children) < 1 or \
                    current_feature not in current_node.children:
                # ① 当前节点为叶子节点，或者由于数据集不充分当前节点的孩子节点不存在下一个划分节点的某一个取值
                # ② 例如根据测试数据集load_simple_data（）构造得到的id3树，对于特征['0','1','D']来说，
                # ③ 在遍历最后一个特征维度时，取值'D'就不存在于孩子节点中
                return current_node.values
            current_node = current_node.children[current_feature]

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


def load_simple_data():
    x = np.array([['0', '0', '0', '0', '0', '0', '0', '1', '1', '1', '1', '1', '1', '1', '1'],
                  ['1', '1', '1', '0', '1', '0', '0', '0', '0', '0', '1', '1', '1', '0', '0'],
                  ['T', 'S', 'S', 'T', 'T', 'T', 'D', 'T', 'T', 'D', 'D', 'T', 'T', 'S', 'S']]).transpose()
    y = np.array([1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1])
    return x, y


def test_compute():
    x, y = load_simple_data()
    dt = DecisionTree()
    ety = dt._compute_entropy(y)
    print(f"信息熵{ety}")
    for i in range(x.shape[1]):
        print(f"条件熵为{dt._compute_condition_entropy(x[:, i], y)}")


def test_decision_tree():
    x, y = load_simple_data()
    dt = DecisionTree(criterion='c45')
    dt.fit(x, y)
    dt.level_order()
    y_pred = dt.predict(np.array([['0', '0', 'T'],
                                  ['0', '1', 'S'],
                                  ['0', '1', 'D'],
                                  ['0', '1', 'T']]))
    logging.info(f"DecisionTree 预测结果为：{y_pred}")


def load_data():
    x, y = load_cut_spam()
    x_train, x_test, y_train, y_test \
        = train_test_split(x, y, test_size=0.3, random_state=2020)
    vect = VectWithoutFrequency(top_k_words=1000)
    x_train = vect.fit_transform(x_train)
    x_test = vect.transform(x_test)
    return x_train, x_test, y_train, y_test


def test_spam_classification():
    x_train, x_test, y_train, y_test = load_data()
    dt = DecisionTree(criterion="id3")
    dt.fit(x_train, y_train)
    y_pred = dt.predict(x_test)
    logging.info(f"DecisionTree 准确率：{accuracy_score(y_test, y_pred)}")
    # logging.info(f"DecisionTree 预测结果为：{y_pred}")
    model = DecisionTreeClassifier(criterion='entropy')
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    logging.info(f"DecisionTreeClassifier 准确率：{accuracy_score(y_test, y_pred)}")


def test_decision_tree_pruning():
    x, y = load_simple_data()
    dt = DecisionTree(criterion='id3', alpha=0.)
    dt.fit(x, y)
    x_test = np.array([['0', '1', 'S'],
                       ['0', '1', 'T']])
    logging.debug(f"剪枝前的层次遍历结果")
    # dt.level_order()
    logging.info(f"剪枝前的预测类别：{dt.predict(x_test)}")

    dt = DecisionTree(criterion='id3', alpha=1.25)
    dt.fit(x, y)
    logging.debug(f"剪枝后的层次遍历结果")
    # dt.level_order()
    logging.info(f"剪枝后的预测类别：{dt.predict(x_test)}")


if __name__ == '__main__':
    formatter = '[%(asctime)s] - %(levelname)s: %(message)s'
    logging.basicConfig(level=logging.INFO,  # 如果需要查看简略信息可将该参数改为logging.INFO
                        format=formatter,  # 关于Logging模块的详细使用可参加文章 https://mp.weixin.qq.com/s/cvO6hCiHMJqC4-4AuUlydw
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[logging.StreamHandler(sys.stdout)])
    test_compute()
    # test_decision_tree()
    # test_spam_classification()  # Accuracy:  id3:0.977  c45 0.975
    # test_decision_tree_pruning()
