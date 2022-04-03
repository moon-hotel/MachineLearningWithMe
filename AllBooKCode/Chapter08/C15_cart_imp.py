import numpy as np
import logging
import sys
import copy
from sklearn.datasets import load_iris
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
               f"当前节点状态时划分特征离散化区间为 {self.split_range}\n"


class CART(object):
    def __init__(self, min_samples_split=2,
                 epsilon=1e-5):
        self.root = None
        self.min_samples_split = min_samples_split  # 用来控制是否停止分裂
        self.epsilon = epsilon  # 停止标准

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
        for i in range(len(feature_values) - 1):  # 遍历当前特征维度中，用每个特征取值将样本划分为两个部分时的基尼指数
            index = (feature_values[i] <= x_feature) & \
                    (x_feature <= feature_values[i + 1])  # 根据当前特征维度的取值，来取对应的特征维度
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
        logging.debug(f"当前节点中的样本基尼指数为 {gini}")
        min_gini = 99999
        split_id = None  # 保存所有可用划分特征中，能够值得基尼指数最小的特征 对应特征离散区间的起始索引
        split_sample_idx = None  # 最小基尼指数下对应的样本划分索引
        best_feature_id = -1  # 保存所有可用划分特征中，能够值得基尼指数最小的特征 对应的特征ID
        for f_id in f_ids:  # 遍历每个特征
            # 遍历特征下的每种取值方式的基尼指数，并返回最小的
            m_gini, s_id, s_s_idx = self._compute_gini_da(f_id, data)
            if m_gini < min_gini:  # 查找所有特征所有取值方式下，基尼指数最小的
                min_gini = m_gini
                split_id = s_id
                split_sample_idx = s_s_idx
                best_feature_id = f_id
        if min_gini < self.epsilon:
            return node
        node.feature_id = best_feature_id
        feature_values = self.feature_values[best_feature_id]
        node.split_range = [feature_values[split_id], feature_values[split_id + 1]]
        logging.debug(f"【***此时选择第{best_feature_id}个特征进行样本划分，"
                      f"此时第{best_feature_id}个特征对应的离散化特征取值范围为 {node.split_range}，"
                      f"最小基尼指数为 {min_gini}***】")
        left_data = data[split_sample_idx]
        right_data = data[~split_sample_idx]
        candidate_ids = copy.copy(f_ids)
        candidate_ids.remove(best_feature_id)  # 当前节点划分后的剩余特征集
        if len(left_data) > 0:
            node.left_child = self._build_tree(left_data, candidate_ids)  # 递归构建决策树
        if len(right_data) > 0:
            node.right_child = self._build_tree(right_data, candidate_ids)
        return node

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
        self.feature_values = self._get_feature_values(X)  # 得到离散化特征
        self._X = np.hstack(([X, np.arange(len(X)).reshape(-1, 1)]))
        # 将训练集中每个样本的序号加入到X的最后一列
        self._build_tree(self._X, feature_ids)  # 递归构建决策树

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
                if node.left_child:
                    queue.append(node.left_child)
                if node.right_child:
                    queue.append(node.right_child)
            res.append(tmp)
        if return_node:
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
            if current_node.split_range is None:
                # ① 当前节点为叶子节点
                return current_node.values
            current_feature_id = current_node.feature_id
            current_feature = x[current_feature_id]
            split_range = current_node.split_range
            if split_range[0] <= current_feature <= split_range[1]:
                current_node = current_node.left_child
            else:
                current_node = current_node.right_child

            #
            # exist_child = False
            # for i in range(len(current_feature_values) - 1):
            #     if current_feature_values[i] <= current_feature <= current_feature_values[i + 1]:
            #         exist_child = True
            #         if str(current_feature_values[i + 1]) not in current_node.children:
            #             # 由于数据集不充分当前节点的孩子节点不存在下一个划分节点的某一个取值
            #             # 例如根据测试数据集load_simple_data（）构造得到的id3树，对于特征[0,1,0]来说，
            #             # 在遍历最后一个特征维度时，取值0就不存在于孩子节点中
            #             return current_node.values
            #         current_node = current_node.children[str(current_feature_values[i + 1])]
            # if not exist_child:
            #     return current_node.values

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
                                  [0, 1, 0],
                                  [0, 1, 2]]))
    logging.info(f"CART 预测结果为：{y_pred}")


def test_iris_classification():
    x, y = load_iris(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2022)
    dt = CART(min_samples_split=2)
    dt.fit(x_train, y_train)
    y_pred = dt.predict(x_test)
    logging.info(f"CART 准确率：{accuracy_score(y_test, y_pred)}")

    model = DecisionTreeClassifier(criterion='gini')
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    logging.info(f"DecisionTreeClassifier 准确率：{accuracy_score(y_test, y_pred)}")


if __name__ == '__main__':
    formatter = '[%(asctime)s] - %(levelname)s: %(message)s'
    logging.basicConfig(level=logging.DEBUG,  # 如果需要查看简略信息可将该参数改为logging.INFO
                        format=formatter,  # 关于Logging模块的详细使用可参加文章https://www.ylkz.life/tools/p10958151/
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[logging.StreamHandler(sys.stdout)])
    # test_gini()
    # test_cart()
    test_iris_classification()
