import numpy as np
from sklearn.preprocessing import LabelBinarizer


def compute_class_prior(y, sample_weight=None):
    """
    计算先验概率
    :return:
    """
    labelbin = LabelBinarizer()  # 将标签转化为one-hot形式
    Y = labelbin.fit_transform(y)  # one-hot 形式标签 shape: [n,n_classes]
    print(Y)
    if sample_weight is not None:  # 每个样本对应的权重，只是在Boost方法中被作为基分类器（weak learner）时用到该参数
        Y = Y.astype(np.float64, copy=False)  # 相关原理将在第8章AdaBoost算法中进行介绍
        sample_weight = np.reshape(sample_weight, [1, -1])  # [1,n_samples]
        Y *= sample_weight.T  # [n_samples,n_classes_] * [n_samples,1] 按位乘
    class_count = Y.sum(axis=0)  # Y: shape(n,n_classes)   Y.sum(): shape(n_classes,)
    class_prior = class_count / class_count.sum()
    return class_prior


if __name__ == '__main__':
    y = np.array([0, 1, 1, 2, 2])
    sample_weight = np.array([0.1, 0.1, 0.3, 0.4, 0.1])
    # sample_weight = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
    class_prior = compute_class_prior(y)
    print(f"不考虑样本权重时的先验概率：{class_prior}")
    class_prior = compute_class_prior(y, sample_weight)
    print(f"考虑样本权重时的先验概率：{class_prior}")
