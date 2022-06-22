import logging
from copy import deepcopy
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import train_test_split
import numpy as np


def kernel(X, y=None, gamma=None):
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    K = euclidean_distances(X, y, squared=True)
    K *= -gamma
    np.exp(K, K)  # <==> K = np.exp(K)
    return K


class LabelPropagation(object):
    """Label Propagation classifier
    """

    def __init__(self, gamma=20., max_iter=1000, tol=1e-3):
        self.gamma = gamma
        self.max_iter = max_iter  # 最大迭代次数
        self.tol = tol  # 误差容忍度

    def _get_kernel(self, X, y=None):
        # 注意，这里的y不是指的训练标签，而是另外一个矩阵
        # 当 y为None时（此时X为训练样本），即模型训练时，建立训练样本中各个样本点之间距离关系，得到矩阵W
        # 此时计算的是X中各个样本点两两之间的权重距离
        #
        # 当 y不为None（此时X为训练样本，y为预测样本）即模型预测时，建立测试样本与训练样本之间距离关系，得到矩阵W
        # 此时计算的是X中各个样本与y中各个样本之间的权重距离
        return kernel(X, y, gamma=self.gamma)

    def _build_graph(self):
        """Matrix representing a fully connected graph between each sample

        This basic implementation creates a non-stochastic affinity matrix, so
        class distributions will exceed 1 (normalization may be desired).
        """
        affinity_matrix = self._get_kernel(self.X_)  # [n_samples,n_samples]
        logging.debug(f" ## 计算矩阵W完毕，形状为:{affinity_matrix.shape}")
        normalizer = affinity_matrix.sum(axis=0, keepdims=True)  # [1, n_samples]
        affinity_matrix /= normalizer  # [n_samples,n_samples] 归一化得到矩阵T
        logging.debug(f" ## 计算矩阵T(归一化）完毕，形状为:{affinity_matrix.shape}")
        return affinity_matrix

    def predict(self, X):
        """
        预测
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data matrix.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            Predictions for input data.
        """
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)].ravel()

    def predict_proba(self, X):
        """
        概率预测
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data matrix.

        Returns
        -------
        probabilities : shape (n_samples, n_classes)
        """
        logging.info(f" ## 进入预测阶段")  # [num_unlabeled,classes]
        weight_matrices = self._get_kernel(self.X_, X).T  # [X.shape[0],n_train_sample]
        probabilities = np.matmul(weight_matrices, self.label_distributions_)
        normalizer = np.sum(probabilities, axis=1, keepdims=True)
        probabilities /= normalizer
        return probabilities

    def fit(self, X, y):
        """
        模型拟合
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            A matrix of shape (n_samples, n_samples) will be created from this.

        y : array-like of shape (n_samples,)
            `n_labeled_samples` (unlabeled points are marked as -1)
            All unlabeled samples will be transductively assigned labels.
        """
        unlabeled = y == -1
        X = np.vstack([X[~unlabeled], X[unlabeled]])
        y = y.reshape(-1, 1)
        y = np.vstack([y[~unlabeled], y[unlabeled]]).reshape(-1)
        self.X_ = X
        logging.info(f"### 正在拟合模型……")
        logging.debug(f" ## 建立样本点之间的距离关系")
        self.graph_matrix = self._build_graph()  # 得到矩阵T，[n_samples,n_samples]
        classes = np.unique(y)  # 得到分类类别，例如三分类可能是 [-1,0,1,2]，其中-1表示对应样本无标签
        classes = (classes[classes != -1])
        logging.debug(f" ## 训练集中样本的标签取值为: {classes}")
        self.classes_ = classes
        n_samples, n_classes = len(y), len(classes)  # 得到样本数和类别数
        unlabeled = y == -1  # 得到没有标签的样本对应的索引
        # unlabeled为[True, False, False, False, True]这样的形式，True表示对应样本没有标签
        # 初始化标签分布情况, [n_samples, n_classes]
        self.label_distributions_ = np.zeros((n_samples, n_classes))
        # 用来记录训练集中每个样本的类别所属概率
        logging.debug(f" ## 初始化标签分布")
        for label in classes:  # 遍历每个类别
            self.label_distributions_[y == label, classes == label] = 1
            # 把已经有标签的样本，在对应类别维度上置为1，会类似得到下面这样一个矩阵：
            #  [[0,0,1],[0,0,0],[1,0,0]]  其中2个样本表示没有标签
        num_unlabeled = np.sum(unlabeled)  # 统计无标签样本个数
        num_labeled = len(y) - num_unlabeled  # 统计有标签样本个数
        Tuu = self.graph_matrix[-num_unlabeled:, -num_unlabeled:]  # [num_unlabeled,num_unlabeled]
        Tul = self.graph_matrix[-num_unlabeled:, :num_labeled]  # [num_unlabeled,num_labeled]
        YL = self.label_distributions_[:num_labeled]  # [num_labeled,classes]
        inv = np.linalg.inv(np.eye(num_unlabeled) - Tuu)  # [num_unlabeled,num_nulabeled]
        YU = np.matmul(np.matmul(inv, Tul), YL)  # [num_unlabeled,classes]
        # [num_unlabeled,num_unlabeled] @ [num_unlabeled,num_labeled] @ [num_labeled,classes]
        self.label_distributions_ = np.vstack([YL, YU])
        return self


def load_data():
    x, y = load_iris(return_X_y=True)
    x_train, x_test, y_train, y_test = \
        train_test_split(x, y, test_size=0.3, random_state=2022)
    rng = np.random.RandomState(20)
    random_unlabeled_points = rng.rand(y_train.shape[0]) < 0.8
    y_mixed = deepcopy(y_train)
    y_mixed[random_unlabeled_points] = -1
    return x_train, x_test, y_train, y_test, y_mixed


def test_label_propagation():
    x_train, x_test, y_train, y_test, y_mixed = load_data()
    model = LabelPropagation()
    model.fit(x_train, y_mixed)

    y_pred = model.predict(x_train)
    logging.info(f"模型在训练集上的准确率为: {accuracy_score(y_pred, y_train)}")
    y_pred = model.predict(x_test)
    logging.info(f"模型在测试集上的准确率为: {accuracy_score(y_pred, y_test)}")


if __name__ == '__main__':
    formatter = '[%(asctime)s] - %(levelname)s: %(message)s'
    logging.basicConfig(level=logging.DEBUG,  # 如果需要查看详细信息可将该参数改为logging.DEBUG
                        format=formatter,  # 关于Logging模块的详细使用可参加文章https://www.ylkz.life/tools/p10958151/
                        datefmt='%Y-%m-%d %H:%M:%S', )
    test_label_propagation()
