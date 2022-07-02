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
        logging.info(f" ## 进入预测阶段")
        weight_matrices = self._get_kernel(self.X_, X).T  # [X.shape[0],n_train_sample]
        probabilities = np.matmul(weight_matrices, self.label_distributions_)
        normalizer = np.sum(probabilities, axis=1, keepdims=True)
        normalizer[normalizer == 0] = 1  # 优化当normalizer为0时的情况
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
        self.X_ = X
        logging.info(f"### 正在拟合模型……")
        logging.debug(f" ## 建立样本点之间的距离关系")
        self.graph_matrix = self._build_graph()  # 得到矩阵T，[n_samples,n_samples]
        classes = np.unique(y)  # 得到分类类别，例如三分类可能是 [-1,0,1,2]，其中-1表示对应样本无标签
        self.classes_ = (classes[classes != -1])
        logging.debug(f" ## 训练集中样本的标签取值为: {self.classes_}")
        n_samples, n_classes = len(y), len(self.classes_)  # 得到样本数和类别数
        unlabeled = y == -1  # 得到没有标签的样本对应的索引
        unlabeled = np.reshape(unlabeled, (-1, 1))
        # unlabeled为[True, False, False, False, True]这样的形式，True表示对应样本没有标签
        # 初始化标签分布情况, [n_samples, n_classes]
        self.label_distributions_ = np.zeros((n_samples, n_classes))
        # 用来记录训练集中每个样本的类别所属概率
        logging.debug(f" ## 初始化标签分布")
        for label in self.classes_:  # 遍历每个类别
            self.label_distributions_[y == label, self.classes_ == label] = 1
            # 把已经有标签的样本，在对应类别维度上置为1，会类似得到下面这样一个矩阵：
            #  [[0,0,1],[0,0,0],[1,0,0]]  其中2个样本表示没有标签
        y_static = np.copy(self.label_distributions_)
        l_previous = np.zeros((n_samples, n_classes))
        logging.info(f" ## 进入迭代阶段")
        for self.n_iter_ in range(self.max_iter):
            if np.abs(self.label_distributions_ - l_previous).sum() < self.tol:
                break
            l_previous = self.label_distributions_
            # 根据公式Y=TY计算得到Y [n_samples,n_samples]  @ [n_samples, n_classes] = [n_samples, n_classes]
            self.label_distributions_ = np.matmul(self.graph_matrix, self.label_distributions_)
            normalizer = np.sum(self.label_distributions_, axis=1, keepdims=True)
            normalizer[normalizer == 0] = 1  # 优化当normalizer为0时的情况
            self.label_distributions_ /= normalizer  # 进行标准化
            self.label_distributions_ = np.where(unlabeled, self.label_distributions_, y_static)
            # 在self.label_distributions_中对满足条件的（unlabeled里面为FALSE）的位置，用y_static中对应位置上的值去替换
            # 也就是将y_static中，原始有正确标签的标签替换到self.label_distributions_中的对应位置
            # a = np.array([1, 2, 3, 5, 6])
            # u = np.array([True, True, False, False, True])
            # c = np.array([1, 1, 99, 99, 1])
            # b = np.where(u, a, c) # b= [ 1  2 99 99  6]
        else:
            logging.warning('max_iter=%d was reached without convergence.' % self.max_iter)
        normalizer = np.sum(self.label_distributions_, axis=1, keepdims=True)
        normalizer[normalizer == 0] = 1
        self.label_distributions_ /= normalizer  # 进行标准化
        self.transduction_ = self.classes_[np.argmax(self.label_distributions_, axis=1)]
        return self

    def score(self, X, y):
        return accuracy_score(self.predict(X), y)


def test_compute_W_and_T():
    x = np.array([[0.5, 0.5], [1.5, 1.5], [2.5, 1.5], [1.5, 2.5], [3.5, 2.5]])
    y = np.array([0, 1, 1, -1, -1])
    W = kernel(x, gamma=1.0)
    model = LabelPropagation(gamma=1.)
    model.fit(x, y)
    T = model.graph_matrix
    logging.info(f"W: \n{np.round(W, 3)}")
    logging.info(f"T: \n{np.round(T, 3)}")


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

    logging.info(f"模型在训练集上的准确率为: {model.score(x_train, y_train)}")
    logging.info(f"模型在测试集上的准确率为: {model.score(x_test, y_test)}")


if __name__ == '__main__':
    formatter = '[%(asctime)s] - %(levelname)s: %(message)s'
    logging.basicConfig(level=logging.DEBUG,  # 如果需要查看详细信息可将该参数改为logging.DEBUG
                        format=formatter,  # 关于Logging模块的详细使用可参加文章https://www.ylkz.life/tools/p10958151/
                        datefmt='%Y-%m-%d %H:%M:%S', )
    test_compute_W_and_T()
    test_label_propagation()
