from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score


def load_data():
    data = load_iris()
    x, y = data.data, data.target
    return x, y


def train(x, y, K):
    model = KMeans(n_clusters=K)
    model.fit(x)
    y_pred = model.predict(x)
    nmi = normalized_mutual_info_score(y, y_pred)
    print("NMI: ", nmi)


if __name__ == '__main__':
    x, y = load_data()
    train(x, y, K=3)
