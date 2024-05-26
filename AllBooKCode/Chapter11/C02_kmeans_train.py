from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score


def load_data():
    data = load_iris()
    x, y = data.data, data.target
    return x, y


def train(x, y, K):
    model = KMeans(n_clusters=K)
    model.fit(x)
    y_pred = model.predict(x)
    ari = adjusted_rand_score(y, y_pred)
    print("ARI: ", ari)


if __name__ == '__main__':
    x, y = load_data()
    train(x, y, K=3)
