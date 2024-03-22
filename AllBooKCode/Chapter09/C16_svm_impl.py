import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.datasets import load_iris
from sklearn.svm import SVC


def kernel_linear(X, x):
    """
    线性核函数
    :param X: [n_samples,n_features] 或 [n_features,]
    :param x: [n_features,]
    :return: shape: [n_features]
    """
    return np.dot(X, x)


def kernel_rbf(X, x):
    """
    高斯核函数 K(X,x)=\exp{\left(\frac{-||X-x||^2}{2\sigma^2}\right)}
    :param X: [n_samples,n_features] 或 [n_features,]
    :param x: [n_features,]
    :param sigma: float
    :return: shape [n_samples, ]
    """
    if X.ndim > 1:
        sigma = 1 / X.shape[1]
    else:
        sigma = 1 / X.shape[0]
    k = -0.5 * np.sum((X - x) ** 2, axis=-1) / sigma ** 2
    return np.exp(k)


def f_x(X, y, alphas, x, b, kernel):
    """
    :param X:  shape [n_samples, n_features]
    :param y:  shape [n_samples,]
    :param alphas:  shape [n_samples,]
    :param x:       shape [n_features,]
    :param b:       shape (1,)
    :return:
    """
    k = kernel(X, x)
    r = alphas * y * k
    return np.sum(r) + b


def compute_eta(x_1, x_2, kernel):
    return kernel(x_1, x_1) - 2 * kernel(x_1, x_2) + kernel(x_2, x_2)


def compute_E_i(f_x_i, y_i):
    return f_x_i - y_i


def compute_alpha_2(alpha_2, E_1, E_2, y_2, eta):
    return alpha_2 + (y_2 * (E_1 - E_2) / eta)


def compute_L_H(C, alpha_1, alpha_2, y_1, y_2):
    L = np.max((0., alpha_2 - alpha_1))
    H = np.min((C, C + alpha_2 - alpha_1))
    if y_1 == y_2:
        L = np.max((0., alpha_1 + alpha_2 - C))
        H = np.min((C, alpha_1 + alpha_2))
    return L, H


def clip_alpha_2(alpha_2, H, L):
    if alpha_2 > H:
        return H
    if alpha_2 < L:
        return L
    return alpha_2


def compute_alpha_1(alpha_1, y_1, y_2, alpha_2, alpha_2_old):
    return alpha_1 + y_1 * y_2 * (alpha_2_old - alpha_2)


def compute_b1(b, E_1, y_1, alpha_1, alpha_1_old,
               x_1, y_2, alpha_2, alpha_2_old, x_2, kernel):
    p1 = b - E_1 - y_1 * (alpha_1 - alpha_1_old) * kernel(x_1, x_1)
    p2 = y_2 * (alpha_2 - alpha_2_old) * kernel(x_1, x_2)
    return p1 - p2


def compute_b2(b, E_2, y_1, alpha_1, alpha_1_old,
               x_1, x_2, y_2, alpha_2, alpha_2_old, kernel):
    p1 = b - E_2 - y_1 * (alpha_1 - alpha_1_old) * kernel(x_1, x_2)
    p2 = y_2 * (alpha_2 - alpha_2_old) * kernel(x_2, x_2)
    return p1 - p2


def clip_b(alpha_1, alpha_2, b1, b2, C):
    if alpha_1 > 0 and alpha_1 < C:
        return b1
    if alpha_2 > 0 and alpha_2 < C:
        return b2
    return (b1 + b2) / 2


def select_j(i, m):
    j = np.random.randint(m)
    while i == j:
        j = np.random.randint(m)
    return j


def smo(C, tol, max_passes, data_x, data_y, kernel):
    """
    SMO求解步骤实现
    :param C:惩罚系数
    :param tol: 误差容忍度
    :param max_passes:当alpha_i不再发生变化时继续迭代更新的最大次数;
    :param data_x: 训练集特征 [n_samples,n_features]
    :param data_y: 训练集标签 [n_samples,]
    :param kernel: 核函数
    :return:
    """
    m, n = data_x.shape
    b, passes = 0., 0
    alphas = np.zeros(shape=(m))
    alphas_old = np.zeros(shape=(m))
    while passes < max_passes:
        num_changed_alphas = 0
        for i in range(m):
            x_i, y_i, alpha_i = data_x[i], data_y[i], alphas[i]
            f_x_i = f_x(data_x, data_y, alphas, x_i, b, kernel)
            E_i = compute_E_i(f_x_i, y_i)
            if ((y_i * E_i < -tol and alpha_i < C) or (y_i * E_i > tol and alpha_i > 0.)):
                j = select_j(i, m)
                x_j, y_j, alpha_j = data_x[j], data_y[j], alphas[j]
                f_x_j = f_x(data_x, data_y, alphas, x_j, b, kernel)
                E_j = compute_E_i(f_x_j, y_j)
                alphas_old[i], alphas_old[j] = alpha_i, alpha_j
                L, H = compute_L_H(C, alpha_i, alpha_j, y_i, y_j)
                if L == H:
                    continue
                eta = compute_eta(x_i, x_j, kernel)
                if eta <= 0:
                    continue
                alpha_j = compute_alpha_2(alpha_j, E_i, E_j, y_j, eta)
                alpha_j = clip_alpha_2(alpha_j, H, L)
                alphas[j] = alpha_j
                if np.abs(alpha_j - alphas_old[j]) < 10e-5:
                    continue
                alpha_i = compute_alpha_1(alpha_i, y_i, y_j, alpha_j, alphas_old[j])
                alphas[i] = alpha_i
                b1 = compute_b1(b, E_i, y_i, alpha_i, alphas_old[i],
                                x_i, y_i, alpha_j, alphas_old[j], x_j, kernel)
                b2 = compute_b2(b, E_j, y_i, alpha_i, alphas_old[i],
                                x_i, x_j, y_j, alpha_j, alphas_old[j], kernel)
                b = clip_b(alpha_i, alpha_j, b1, b2, C)
                num_changed_alphas += 1

        if num_changed_alphas == 0:
            passes += 1
        else:
            passes = 0
    return alphas, b


class SVM(object):
    def __init__(self, C=1., tol=0.001, kernel='rbf', max_passes=20):
        self.C = C  # 惩罚项系数
        self.tol = tol  # 裁剪alpha时的容忍度
        self.max_passes = max_passes  # 当alpha不再发生变化时继续迭代更新的最大次数;
        if kernel == 'rbf':
            self.kernel = kernel_rbf
        elif kernel == 'linear':
            self.kernel = kernel_linear
        else:
            raise ValueError(f"核函数{kernel}未实现")

    def fit(self, X, y):
        """
        拟合SVM分类器
        :param X:  shape: [n_samples,n_features]
        :param y:  shape: [n_samples,]
        :return:
        """
        self._X = X
        self.alphas = []  # 用来保存每个二分类器计算得到的alpha参数，因为在多分类问题中
        self.bias = []  # 采用的是ovr策略； bias用来保存每个分类器对应的偏置
        labelbin = LabelBinarizer(neg_label=-1)  # 将标签转化为one-hot形式
        Y = labelbin.fit_transform(y)  # one-hot 形式标签 shape: [n_samples,n_classes]
        self.classes_ = labelbin.classes_  # 原始标签类别 shape: [n_classes,]
        if Y.shape[1] == 1:  # 当数据集为二分类时fit_transform处理后的结果并不是one-hot形式
            Y = np.concatenate((-1 * Y, Y), axis=1)  # 改变为one-hot形式
        self.n_classes = Y.shape[1]  # 数据集的类别数量
        self.Y = Y
        for c in range(self.n_classes):
            self._fit_binary(self._X, Y[:, c])  # 分别为每个类别拟合得到一个二分类器
        self.alphas = np.vstack((self.alphas))  # [n_classes，n_samples]
        self.bias = np.array(self.bias)  # [n_classes,]

    def _fit_binary(self, X, y):
        """
        拟合二分类SVM
        :param X: shape: [n_samples,n_features], 原始训练特征
        :param y: shape: [n_samples,]， 只含有 -1 和 +1的标签后特征，用来拟合一个二分类器
        :return:
        """
        alphas, bias = smo(C=self.C, tol=self.tol,
                           max_passes=self.max_passes,
                           data_x=X, data_y=y, kernel=self.kernel)
        self.alphas.append(alphas)
        self.bias.append(bias)

    def predict_one_sample(self, x, y, alphas, bias):
        """
        根据公式 \hat{y}=\sum_{i=1}^m\alpha_iy^{(i)}\langle \phi(x^{(i)}),\phi(x)\rangle+b
        来预测样本类别
        :param x: shap: [n_features,]  新输入样本
        :param y: [n_samples,]， 训练集中只含有 -1 和 +1的标签化特征
        :param alphas: # 当前分类器对应的alphas参数
        :param bias:
        :return:
        """
        y_score = np.sum(y * self.kernel(self._X, x) * alphas) + bias
        return y_score

    def _predict_binary(self, X, y, alphas, bias):
        """
        二分类预测
        :param X: 预测样本 shape: [n_samples,n_features]
        :param y: 每个类别对应的标签: [n_samples,]
        :param alphas: [n_samples, ]
        :param bias: [1,]
        :return: [n_samples,]
        """
        y_scores = []
        for x in X:
            y_scores.append(self.predict_one_sample(x, y, alphas, bias))
        return y_scores

    def predict(self, X, return_prob=False):
        """
        预测函数
        :param X: shape: [n_samples,n_features]
        :param return_prob:  是否返回概率
        :return:
        """
        all_y_scores = []
        for c in range(self.n_classes):
            y_scores = self._predict_binary(X, self.Y[:, c], self.alphas[c], self.bias[c])
            all_y_scores.append(y_scores)
        all_y_scores = np.vstack((all_y_scores)).transpose()  # [n_samples,n_classes]
        prob = np.exp(all_y_scores) / np.sum(np.exp(all_y_scores), 1, keepdims=True)
        y_pred = np.argmax(prob, axis=-1)
        if return_prob:
            return y_pred, prob
        return y_pred


def load_data():
    x, y = load_iris(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2022)  # 26
    ss = StandardScaler()
    x_train = ss.fit_transform(x_train)
    x_test = ss.transform(x_test)
    return x_train, x_test, y_train, y_test


def test_simple_binary_classification():
    X = np.array([[5, 1], [0, 2], [1, 5], [3., 2],
                  [1, 2], [3, 5], [1.5, 6], [4.5, 6], [0, 7]])
    y = np.array([0, 0, 0, 0,
                  0, 1, 1, 1, 1])
    x_test = np.array([[1, 0], [2, 6]])
    model = SVM(C=1.5, tol=0.1, max_passes=200)
    model.fit(X, y)
    y_pred = model.predict(x_test)
    print(y_pred)


def test_iris_classification():
    x_train, x_test, y_train, y_test = load_data()
    model = SVM(C=1., tol=0.001, max_passes=20, kernel='rbf')
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print("手动实现准确率：", accuracy_score(y_pred, y_test))

    model = SVC(C=1, kernel='rbf')
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print("sklean上准确率：", accuracy_score(y_pred, y_test))


if __name__ == '__main__':
    test_simple_binary_classification()
    test_iris_classification()

    # [0 1]
    # 手动实现准确率： 0.9777777777777777
    # sklean上准确率： 0.9777777777777777
