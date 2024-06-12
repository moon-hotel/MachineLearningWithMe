"""
文件名: AllBooKCode/Chapter07/C01_naive_bayes_category.py
创建时间: 2022年
作者: @空字符
公众号: @月来客栈
知乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import logging
import numpy as np
import sys

sys.path.append('../')
from utils import load_cut_spam
from utils import VectWithoutFrequency


def load_simple_data():
    x = np.array([[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
                  [2, 1, 1, 2, 2, 2, 0, 2, 2, 0, 0, 2, 2, 1, 1]]).transpose()
    y = np.array([1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1])
    return x, y


def load_data():
    x, y = load_cut_spam()
    x_train, x_test, y_train, y_test \
        = train_test_split(x, y, test_size=0.3, random_state=2020)
    vect = VectWithoutFrequency(top_k_words=1000)
    x_train = vect.fit_transform(x_train)
    x_test = vect.transform(x_test)
    return x_train, x_test, y_train, y_test


class MyCategoricalNB(object):
    """
    Parameters:
        alpha: 平滑项，默认为1，即拉普拉斯平滑
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self._ALPHA_MIN = 1e-10

    def _check_alpha(self):
        """
        检查 alpha的取值
        :return:
        """
        if np.min(self.alpha) < self._ALPHA_MIN:
            self.alpha = np.maximum(self.alpha, self._ALPHA_MIN)

    def _init_counters(self):
        """
        初始化计数器
        :return:
        """
        self.class_count_ = np.zeros(self.n_classes, dtype=np.float64)
        # shape: [n_classes, ] 后续用来记录每个类别下的样本数
        # 每个维度表示每个类别的样本数量，e.g. [2,2,3] 表示0,1,2这三个类别的样本数分别是2,2,3
        # 其作用是后续用来计算每个类别的先验概率
        self.category_count_ = [np.zeros((self.n_classes, 0))
                                for _ in range(self.n_features_)]
        # n_features_个元素（array()），目前每个元素的shape是[n_classes,0]
        # 后续每个元素的shape将会更新为[n_classes,len(X_i)], len(X_i)表示X_i这个特征的取值情况数量
        # 目的是用来记录在各个类别下每个特征变量中各种取值情况的数量
        # 例如category_count_[i][j][k]为10 表示含义就是特征i在类别j下特征取值为k的样本数量为10

    def _count(self, X, Y):
        """
        对数据集每个特征维度下的取值情况进行统计
        :param X: shape [n_samples,n_features]
        :param Y: shape [n_samples,]
        :return:
        """

        def _update_cat_count(X_feature, Y, cat_count, n_classes):
            """
            对每一列特征进行统计处理
            :param X_feature:  模型输入的某一列特征X_i, shape: [n_samples,]
            :param Y:    one-hot 形式标签 shape: [n_samples,n_classes]
            :param cat_count:   shape: [n_classes,len(X_i)], len(X_i)表示X_i这个特征的取值情况数量
            :param n_classes:   n_classes,数据集的类别数量
            :return:
            """
            for j in range(n_classes):  # 遍历每个类别
                mask = Y[:, j].astype(bool)  # 取每个类别下对应样本的索引
                counts = np.bincount(X_feature[mask])  # 统计当前类别下，特征X_feature中各个取值下的数量
                # np.bincount的作用的是统计每个值出现的次数，例如
                # counts = np.bincount(np.array([0, 3, 5, 1, 4, 4]))
                # print(counts) [1 1 0 1 2 1]
                # 表示[0, 3, 5, 1, 4, 4]中0,1,2,3,4,5这个6个值的出现的频次分别是1,1,0,1,2,1
                indices = np.nonzero(counts)[0]
                cat_count[j, indices] += counts[indices]
                # cat_count[i,k]表示第i个类别下，特征X_feature第k个取值情况的数量

        self.class_count_ += Y.sum(axis=0)  # Y: shape(n,n_classes)   Y.sum(): shape(n_classes,)
        # self.class_count_的shape是(n_classes,)  每个维度表示每个类别的样本数量
        # e.g. [2,2,3] 表示0,1,2这三个类别的样本数分别是2,2,3
        logging.debug(f"数据集X为:\n{X}")
        logging.debug(f"标签Y为:\n{Y}")
        logging.debug(f"每个类别下的样本数class_count_(n_classes,): {self.class_count_}")
        self.n_categories_ = X.max(axis=0) + 1
        # 统计每个特征维度的 取值数量（因为特征取值是从0开始的所以后面加了1）,e.g.  [3 3 3 3]，表示四个维度的取值均有3中情况
        logging.debug(f"每个特征的取值种数n_categories_:{self.n_categories_}")

        for i in range(self.n_features_):  # 遍历每个特征
            X_feature = X[:, i]  # 取每一列的特征
            self.category_count_[i] = np.pad(self.category_count_[i],
                                             [(0, 0), (0, self.n_categories_[i])],
                                             'constant')  # shape: [n_classes,n_categories_[i]]
            # np.pad(a,((1,2),(3,4)),'constant') 含义是在a的第一个维度（行）的上面和下面各填充1行和2行0，
            # 在a的第二个维度（列）的左边和右边各填充3列和4列0
            # 在原始category_count_[i]的基础上，追加n_categories_[i]列全为0的值，
            # 因为category_count_[i]初始化式时的shape为[n_classes,0]
            _update_cat_count(X_feature, Y,
                              self.category_count_[i],
                              self.n_classes)
        # category_count_为一个包含有n_features个元素的列表
        # category_count_[i][j][k]为10 表示含义就是特征i个在类别j下特征取值为k的样本数量为10
        logging.debug(f"各个特征每个取值的数量分布（未平滑处理） category_count_:\n {self.category_count_}")

    def _update_feature_prob(self):
        """
        计算条件概率
        :return:
        """
        feature_prob = []
        for i in range(self.n_features_):  # 遍历 每一个特征

            # 以下两行是sklearn中的平滑处理方式
            # smoothed_cat_count = self.category_count_[i] + self.alpha  # 平滑处理
            # smoothed_class_count = smoothed_cat_count.sum(axis=1)
            # 以下两行是文中的平滑处理方式
            smoothed_cat_count = self.category_count_[i] + self.alpha
            smoothed_class_count = self.category_count_[i].sum(axis=1) + self.category_count_[i].shape[1] * self.alpha

            cond_prob = smoothed_cat_count / smoothed_class_count.reshape(-1, 1)
            feature_prob.append(cond_prob)
            logging.debug(f"第{i}个特征在各类别下各个特征取值的条件概率为: \n{cond_prob}")
            logging.debug(f"第{i}个特征在各类别下各个特征取值数为: \n{smoothed_cat_count}")
        self.feature_prob_ = feature_prob
        # feature_prob_ 为一个包含有n_features_个元素的列表，每个元素的shape为 (self.n_classes,特征取值数)

    def _update_class_prior(self):
        """
        计算先验概率
        :return:
        """
        logging.debug(f"n_classes:{self.n_classes}")
        logging.debug(f"class_count_:{self.class_count_}")
        # empirical prior, with sample_weight taken into account
        self.class_prior_ = (self.class_count_ + self.alpha) / (self.class_count_.sum() + self.n_classes * self.alpha)
        logging.debug(f"计算每个类别的先验概率class_prior_:{self.class_prior_}")

    def _joint_likelihood(self, X):
        """
        计算后验概率
        :param X: shape: [n_samples,n_features]
        :return:
        """

        if not X.shape[1] == self.n_features_:
            raise ValueError("Expected input with %d features, got %d instead"
                             % (self.n_features_, X.shape[1]))
        jll = np.ones((X.shape[0], self.class_count_.shape[0]))  # 用来累积条件概率
        for i in range(self.n_features_):
            indices = X[:, i]  # 取对应的每一列特征
            if self.feature_prob_[i].shape[1] <= indices.max():
                raise IndexError(f"测试集中的第{i}个特征维度的取值情况"
                                 f" {indices.max()} 超出了训练集中该维度的取值情况！")
            jll *= self.feature_prob_[i][:, indices].T  # 取每个特征取值下对应的条件概率，并进行累乘
            # feature_prob_[i][:, indices]  表示第i个特征下，取对应特征取值对应的条件概率
            # feature_prob_[i]的shape为 (n_classes,特征取值数),
            # feature_prob_[i][j][k]表示特征[i]在类别j下，取值为k时的概率
        total_ll = jll * self.class_prior_  # 条件概率乘以先验概率即得到后验概率
        return total_ll

    def fit(self, X, y):
        """
        Parameters
        ----------
        X : shape: [n_samples,n_features]
        y : shape [n_samples,]
        """
        self.n_features_ = X.shape[1]
        labelbin = LabelBinarizer()  # 将标签转化为one-hot形式
        Y = labelbin.fit_transform(y)  # one-hot 形式标签 shape: [n,n_classes]
        self.classes_ = labelbin.classes_  # 原始标签类别 shape: [n_classes,]
        if Y.shape[1] == 1:  # 当数据集为二分类时fit_transform处理后的结果并不是one-hot形式
            Y = np.concatenate((1 - Y, Y), axis=1)  # 改变为one-hot形式
        self.n_classes = Y.shape[1]  # 数据集的类别数量
        self._init_counters()  # 初始化计数器
        self._count(X, Y)  # 对各个特征的取值情况进行计数，以计算条件概率等
        self._check_alpha()  # 检查平滑
        self._update_class_prior()
        self._update_feature_prob()
        return self

    def predict(self, X, with_prob=False):
        """
        极大化概率进行预测
        Parameters
        ----------
        X : shape: [n_samples,n_features]
        """
        from scipy.special import softmax
        jll = self._joint_likelihood(X)
        logging.debug(f"样本预测原始概率为：{jll}")
        y_pred = self.classes_[np.argmax(jll, axis=1)]
        if with_prob:
            prob = softmax(jll)
            return y_pred, prob
        return y_pred


def test_naive_bayes():
    x, y = load_simple_data()
    logging.info(f"My Bayes 运行结果：")
    model = MyCategoricalNB(alpha=0)
    model.fit(x, y)
    logging.info(model.predict(np.array([[0, 1, 0]]), with_prob=True))
    logging.info(f"CategoricalNB 运行结果：")
    model = CategoricalNB(alpha=0)
    model.fit(x, y)
    logging.info(model.predict(np.array([[0, 1, 0]])))
    logging.info(model.predict_proba(np.array([[0, 1, 0]])))


def test_spam_classification():
    x_train, x_test, y_train, y_test = load_data()
    model = MyCategoricalNB(alpha=1.0)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    logging.info(f"MyCategoricalNB 运行结果：")
    logging.info(classification_report(y_test, y_pred))

    model = CategoricalNB()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    logging.info(f"CategoricalNB 运行结果：")
    logging.info(classification_report(y_test, y_pred))


if __name__ == '__main__':
    formatter = '[%(asctime)s] - %(levelname)s: %(message)s'
    logging.basicConfig(level=logging.DEBUG,  # 如果需要查看详细信息可将该参数改为logging.DEBUG
                        format=formatter,  # 关于Logging模块的详细使用可参加文章https://www.ylkz.life/tools/p10958151/
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[logging.StreamHandler(sys.stdout)]
                        )

    # test_naive_bayes()
    test_spam_classification()