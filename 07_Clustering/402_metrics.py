import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


class Metrics():
    y = []
    y_pre = []

    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def getFscAccNmiAri(self):
        y_true = self.y_true
        y_pred = self.y_pred
        Fscore, Accuracy = self.getFscoreAndAcc()
        NMI = normalized_mutual_info_score(y_true, y_pred)
        ARI = adjusted_rand_score(y_true, y_pred)
        return Fscore, Accuracy, NMI, ARI

    def getFscoreAndAcc(self):
        y_true = np.array(self.y_true)
        y_pred = np.array(self.y_pred)
        n = len(y_pred)  # 样本总数
        assert len(y_pred) == len(y_true), "len(pred) not equal to len(y_true)"
        true = np.unique(y_true)
        pred = np.unique(y_pred)
        true_size = len(true)  # 真实值中有多少个类别
        pred_size = len(pred)  # 预测值中有多少个类别

        # 计算得到混淆矩阵
        a = np.ones((true_size, 1), dtype=int) * y_true  #
        b = true.reshape(true_size, 1) * np.ones((1, n), dtype=int)
        pid = (a == b) * 1  # true_size by n
        a = np.ones((pred_size, 1), dtype=int) * y_pred
        b = pred.reshape(pred_size, 1) * np.ones((1, n), dtype=int)
        cid = (a == b) * 1  # pred_size by n
        confusion_matrix = np.matmul(pid, cid.T)  # 计算得到混淆矩阵
        #        P    N
        #   P   TP   FN
        #   N   FP   TN
        # 计算得到准确率
        temp = np.max(confusion_matrix, axis=0)  # 选择正确数最多计算准确率
        Accuracy = np.sum(temp, axis=0) / float(n)

        # ------------------------计算得到f-score
        ci = np.sum(confusion_matrix, axis=0)  # [TP+FP,FN+TN]  预测
        pj = np.sum(confusion_matrix, axis=1)  # [TP+FN,FP+TN]  真实
        # 分情况交叉计算精确率
        precision = confusion_matrix / (np.ones((true_size, 1), dtype=float) * ci.reshape(1, len(ci)))
        # 分情况交叉计算召回率
        recall = confusion_matrix / (pj.reshape(len(pj), 1) * np.ones((1, pred_size), dtype=float))

        F = 2 * precision * recall / (precision + recall)
        F = np.nan_to_num(F)
        temp = (pj / float(pj.sum())) * np.max(F, axis=0)
        Fscore = np.sum(temp, axis=0)
        return (Fscore, Accuracy)



if __name__ == '__main__':
    y_true = [0] * 20 + [1] * 18 + [1] * 57 + [0] * 5
    y_pred = [0] * 38 + [1] * 62

    metrics = Metrics(y_true, y_pred)
    fsc, acc, nmi, ari = metrics.getFscAccNmiAri()
    print("Fscore:{} , Accuracy:{}, NMI:{}, ARI:{}".format(fsc, acc, nmi, ari))
