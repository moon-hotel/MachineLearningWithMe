from sklearn.metrics import  classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import numpy as np

from sklearn.datasets import load_breast_cancer


def load_data():
    data = load_breast_cancer()
    x, y = data.data, data.target
    return x, y





def get_acc_rec_pre_f(y_true, y_pred, beta=1.0):
    (tn, fp), (fn, tp) = confusion_matrix(y_true, y_pred)
    p1, p2 = tp / (tp + fp), tn / (tn + fn)
    r1, r2 = tp / (tp + fn), tn / (tn + fp)
    f_beta1 = (1 + beta ** 2) * p1 * r1 / (beta ** 2 * p1 + r1)
    f_beta2 = (1 + beta ** 2) * p2 * r2 / (beta ** 2 * p2 + r2)
    m_p, m_r, m_f = 0.5 * (p1 + p2), 0.5 * (r1 + r2), 0.5 * (f_beta1 + f_beta2)
    class_count = np.bincount(y_true)
    w1, w2 = class_count[1] / sum(class_count), class_count[0] / sum(class_count)
    w_p, w_r, w_f = w1 * p1 + w2 * p2, w1 * r1 + w2 * r2, w1 * f_beta1 + w2 * f_beta2
    print(f"宏平均： 精确率：{m_p},召回率：{m_r},F值：{m_f}")
    print(f"加权平均：精确率：{w_p},召回率：{w_r},F值：{w_f}")


def train(x, y):
    model = LogisticRegression(multi_class='ovr')
    model.fit(x, y)
    y_pred = model.predict(x)
    print("准确率: ", model.score(x, y))
    get_acc_rec_pre_f(y, y_pred)


if __name__ == '__main__':
    x, y = load_data()
    train(x, y)