import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import auc


def predict(y_scores, threholds):
    return (y_scores >= threholds) * 1


def compute_scores(y_true, y_pred):
    p_score = precision_score(y_true, y_pred)
    r_score = recall_score(y_true, y_pred)
    return p_score, r_score


def compute_ap(recall, precision):
    # \\text{AP} = \\sum_n (R_n - R_{n-1}) P_n
    rp = [item for item in zip(recall, precision)][::-1]  # 按recall升序进行排序
    ap = 0
    for i in range(1, len(rp)):
        ap += (rp[i][0] - rp[i - 1][0]) * rp[i][1]
        # print(f"({rp[i][0]} - {rp[i - 1][0]}) * {rp[i][1]}")
    return ap


def p_r_curve(y_true, y_scores):
    thresholds = sorted(np.unique(y_scores))
    precisions, recalls = [], []
    for thre in thresholds:
        y_pred = predict(y_scores, thre)
        r = compute_scores(y_true, y_pred)
        precisions.append(r[0])
        recalls.append(r[1])
    # 去掉召回率中末尾重复的情况
    last_ind = np.searchsorted(recalls[::-1], recalls[0]) + 1
    precisions = precisions[-last_ind:]
    recalls = recalls[-last_ind:]
    thresholds = thresholds[-last_ind:]
    precisions.append(1)
    recalls.append(0)
    return precisions, recalls, thresholds


if __name__ == '__main__':
    y_true = np.array([0, 0, 0, 1, 1, 0, 1, 1])
    y_scores = np.array([0.5, 0.55, 0.74, 0.65, 0.28, 0.17, 0.3, 0.45])
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    print("sklearn 计算结果：")
    print("precision", precision)
    print("recall", recall)
    print("thresholds", thresholds)
    print("ap", ap)
    print("auc", auc(recall, precision))

    print("\n编码实现 计算结果：")
    precision, recall, thresholds = p_r_curve(y_true, y_scores)
    ap = compute_ap(recall, precision)
    print("precision", precision)
    print("recall", recall)
    print("thresholds", thresholds)
    print("ap", ap)
