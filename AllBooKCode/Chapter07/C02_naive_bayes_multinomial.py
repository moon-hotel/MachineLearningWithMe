from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import logging
import sys

sys.path.append('../')
from utils import load_cut_spam


class MyMultinomialNB(object):

    def __init__(self, alpha=0.):
        self.alpha = alpha
        self._ALPHA_MIN = 1e-10

    def _check_alpha(self):
        """
        检查 alpha的取值
        :return:
        """
        if np.min(self.alpha) < self._ALPHA_MIN:
            self.alpha = np.maximum(self.alpha, self._ALPHA_MIN)

    def _init_counters(self, ):
        self.class_count_ = np.zeros(self.n_classes, dtype=np.float64)
        self.feature_count_ = np.zeros((self.n_classes, self.n_features_),
                                       dtype=np.float64)
        # feature_count_[i][j]表示，对于整个训练集来说，在第i个类别中第j个特征的出现频次
        # 例如feature_count_ = [[ 6. 31. 36.]
        #                      [44. 22. 28.]
        #                      [19. 10. 14.]]
        # feature_count_[1][0]表示在第1个类别下，在所有样本中第0个特征出现的总频次为44

    def _count(self, X, Y):
        """
        进行样本及特征分布统计
        :param X:
        :param Y:
        :return:
        """
        self.class_count_ += Y.sum(axis=0)  # Y: shape(n,n_classes)   Y.sum(): shape(n_classes,)
        # 计算得到每个类别下的样本数量
        self.feature_count_ += np.dot(Y.T, X)  # [n_classes,n] @ [n,n_features_]
        logging.debug(f"各个类别下特征维度的频次为：\n{self.feature_count_}")
        # 计算得到每个类别下，各个特征维度的频次总数

    def _update_feature_prob(self, ):
        """
        计算条件概率
        :return:
        """
        smoothed_fc = self.feature_count_ + self.alpha
        smoothed_cc = smoothed_fc.sum(axis=1)
        self.feature_prob_ = (np.log(smoothed_fc) -
                              np.log(smoothed_cc.reshape(-1, 1)))
        logging.debug(f"各个类别下特征维度的占比为：\n{self.feature_prob_}")

    def _update_class_prior(self):
        """
        计算先验概率
        :return:
        """
        logging.debug(f"每个类别下的样本数class_count_:{self.class_count_}")
        log_class_count = np.log(self.class_count_)
        self.class_prior_ = log_class_count - np.log(self.class_count_.sum())
        logging.debug(f"计算每个类别的先验概率class_prior_:\n{self.class_prior_}")

    def _joint_likelihood(self, X):
        """Calculate the posterior log probability of the samples X"""
        return np.dot(X, self.feature_prob_.T) + self.class_prior_

    def fit(self, X, y, sample_weight=None):
        """
        Parameters
        ----------
        :param X: shape: [n_samples,n_features]
        :param y: shape [n_samples,]
        :param sample_weight: [n_samples,]
        """
        self.n_features_ = X.shape[1]
        labelbin = LabelBinarizer()  # 将标签转化为one-hot形式
        Y = labelbin.fit_transform(y)  # one-hot 形式标签 shape: [n,n_classes]
        self.classes_ = labelbin.classes_  # 原始标签类别 shape: [n_classes,]
        if Y.shape[1] == 1:  # 当数据集为二分类时fit_transform处理后的结果并不是one-hot形式
            Y = np.concatenate((1 - Y, Y), axis=1)  # 改变为one-hot形式
        if sample_weight is not None:  # 每个样本对应的权重，只是在Boost方法中被作为基分类器（weak learner）时用到该参数
            Y = Y.astype(np.float64, copy=False)  # 相关原理将在第8章AdaBoost算法中进行介绍
            sample_weight = np.reshape(sample_weight, [1, -1])  # [1,n_samples]
            Y *= sample_weight.T  # [n_samples,n_classes_] * [n_samples,1] 按位乘
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
        return: shape: [X.shape[0],]
        """
        from scipy.special import softmax
        jll = self._joint_likelihood(X)
        logging.debug(f"样本预测原始概率为：{jll}")
        y_pred = self.classes_[np.argmax(jll, axis=1)]
        if with_prob:
            prob = softmax(jll)
            return y_pred, prob
        return y_pred


def load_simple_data():
    import numpy as np
    x = np.array([[5, 3, 2, 1, 0, 5, 12, 12, 10, 7, 8, 3, 0, 0, 1],
                  [11, 10, 6, 8, 7, 0, 0, 0, 0, 3, 1, 9, 1, 7, 0],
                  [7, 1, 9, 2, 12, 2, 15, 2, 2, 0, 0, 12, 4, 9, 1]]).transpose()
    y = np.array([1, 1, 0, 0, 2, 1, 1, 2, 1, 2, 1, 0, 0, 0, 1])
    return x, y


def load_data():
    x, y = load_cut_spam()
    x_train, x_test, y_train, y_test = \
        train_test_split(x, y, test_size=0.3, random_state=2020)
    vect = TfidfVectorizer(max_features=1000)
    x_train = vect.fit_transform(x_train)
    x_test = vect.transform(x_test)
    return x_train, x_test, y_train, y_test

def test_count():
    x, y = load_simple_data()
    X = x
    labelbin = LabelBinarizer()  # 将标签转化为one-hot形式
    Y = labelbin.fit_transform(y)
    logging.info(f"{Y.shape}")
    logging.info(f"{X.shape}")
    r = np.dot(Y.T,X)
    logging.info(f"np.dot(Y.T,X): \n{r}")
    logging.info(f"np.dot(Y.T,X): \n{Y.T}")
    logging.info(f"np.dot(Y.T,X): \n{X}")
    #                                       [[ 5 11  7]
    #                                        [ 3 10  1]
    #                                        [ 2  6  9]
    #                                        [ 1  8  2]
    #                                        [ 0  7 12]
    # [[0 0 1 1 0 0 0 0 0 0 0 1 1 1 0]       [ 5  0  2]     [[ 6 31 36]
    #  [1 1 0 0 0 1 1 0 1 0 1 0 0 0 1]   @   [12  0 15]  =  [44 22 28]
    #  [0 0 0 0 1 0 0 1 0 1 0 0 0 0 0]]      [12  0  2]      [19 10 14]]
    #                                        [10  0  2]
    #                                        [ 7  3  0]
    #                                        [ 8  1  0]
    #                                        [ 3  9 12]
    #                                        [ 0  1  4]
    #                                        [ 0  7  9]
    #                                        [ 1  0  1]]
    #

def test_naive_bayes():
    x, y = load_simple_data()
    logging.info(f"MyMultinomialNB运行结果：")
    model = MyMultinomialNB(alpha=1.)
    model.fit(x, y)
    logging.info(model.predict(np.array([[17, 25, 39]]), with_prob=True))
    logging.info(f"MultinomialNB 运行结果：")
    model = MultinomialNB(alpha=1.)
    model.fit(x, y)
    logging.info(model.predict(np.array([[17, 25, 39]])))
    logging.info(model.predict_proba(np.array([[17, 25, 39]])))


def test_spam_classification():
    x_train, x_test, y_train, y_test = load_data()
    logging.info(f"MyMultinomialNB 运行结果：")
    model = MyMultinomialNB(alpha=1.)
    model.fit(x_train.toarray(), y_train)
    y_pred = model.predict(x_test.toarray())
    logging.info(classification_report(y_pred, y_test))

    logging.info(f"MultinomialNB 运行结果：")
    model = MultinomialNB(alpha=1.)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    logging.info(classification_report(y_pred, y_test))


if __name__ == '__main__':
    formatter = '[%(asctime)s] - %(levelname)s: %(message)s'
    logging.basicConfig(level=logging.DEBUG,  # 如果需要查看简略信息可将该参数改为logging.INFO
                        format=formatter,  # 关于Logging模块的详细使用可参加文章https://www.ylkz.life/tools/p10958151/
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[logging.StreamHandler(sys.stdout)]
                        )
    test_count()
    test_naive_bayes() # 7.4.4节
    test_spam_classification() # 7.4.5节 基于Multinomial的垃圾邮件分类
