from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score

if __name__ == '__main__':
    y_true = [0, 0, 0, 1, 1, 1]
    y_pred = [0, 0, 1, 1, 2, 2]
    ARI = adjusted_rand_score(y_true,y_pred)
    NMI = normalized_mutual_info_score(y_true,y_pred)
    print("ARI: {}, NMI: {}".format(ARI,NMI))