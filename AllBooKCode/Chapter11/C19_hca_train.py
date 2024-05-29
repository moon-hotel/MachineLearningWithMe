from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score

if __name__ == '__main__':
    n_clusters = 3
    X, y = load_iris(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    model.fit(X)
    print(f"兰德系数为: {adjusted_rand_score(y, model.labels_)}")
