import numpy as np
from sklearn.metrics.cluster import pair_confusion_matrix


def accuracy(labels_true, labels_pred):
    clusters = np.unique(labels_pred)
    labels_true = np.reshape(labels_true, (-1, 1))
    labels_pred = np.reshape(labels_pred, (-1, 1))
    count = []
    for c in clusters:
        idx = np.where(labels_pred == c)[0]
        labels_tmp = labels_true[idx, :].reshape(-1)
        count.append(np.bincount(labels_tmp).max())
    return np.sum(count) / labels_true.shape[0]


def get_rand_index_and_f_measure(labels_true, labels_pred, beta=1.):
    (tn, fp), (fn, tp) = pair_confusion_matrix(labels_true, labels_pred)

    ri = (tp + tn) / (tp + tn + fp + fn)
    ari = 2. * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
    p, r = tp / (tp + fp), tp / (tp + fn)
    f_beta = (1 + beta**2) * (p * r / ((beta ** 2) * p + r))
    return ri, ari, f_beta


if __name__ == '__main__':
    y_pred = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2]
    y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 0, 0]
    purity = accuracy(y_true, y_pred)
    ri, ari, f_beta = get_rand_index_and_f_measure(y_true, y_pred, beta=1.)
    print(f"purity:{purity}\nri:{ri}\nari:{ari}\nf_measure:{f_beta}")
