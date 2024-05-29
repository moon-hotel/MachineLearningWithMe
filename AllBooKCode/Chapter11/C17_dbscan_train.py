from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score

if __name__ == '__main__':
    X, y = make_moons(n_samples=700, noise=0.05, random_state=2020)
    X = StandardScaler().fit_transform(X)
    db = DBSCAN(eps=0.3, min_samples=10)
    db.fit(X)
    print(f"DBSCAN 聚类结果兰德系数为: {adjusted_rand_score(y, db.labels_)}")
