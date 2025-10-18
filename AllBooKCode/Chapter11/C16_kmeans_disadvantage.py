from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def test_moon_circle():
    X, y = make_circles(n_samples=700, noise=0.05, random_state=2022, factor=0.5)
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.7)
    plt.title("环形",fontsize=15)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1, 2, 2)
    X, y = make_moons(n_samples=700, noise=0.05, random_state=2020)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.7)
    plt.title("月牙形",fontsize=15)
    plt.xticks([])
    plt.yticks([])
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.tight_layout()
    plt.show()


def test_circle():
    X, y = make_circles(n_samples=700, noise=0.05, random_state=2022, factor=0.5)
    model = KMeans(n_clusters=2)
    model.fit(X)
    y_pred = model.predict(X)
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.7)
    plt.title("True Distribution")
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1, 2, 2)
    plt.title("Clustered by KMeans")
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, alpha=0.7)
    plt.xticks([])
    plt.yticks([])
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.tight_layout()
    plt.show()


def test_moon():
    X, y = make_moons(n_samples=700, noise=0.05, random_state=2020)
    model = KMeans(n_clusters=2)
    model.fit(X)
    y_pred = model.predict(X)
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.7)
    plt.title("True Distribution")
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1, 2, 2)
    plt.title("Clustered by KMeans")
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, alpha=0.7)
    plt.xticks([])
    plt.yticks([])
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.tight_layout()
    plt.show()


def test_moon_circle_cluster():
    X, y = make_circles(n_samples=700, noise=0.05, random_state=2022, factor=0.5)
    model = KMeans(n_clusters=2)
    model.fit(X)
    y_pred = model.predict(X)
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, alpha=0.7)
    plt.title("环形",fontsize=15)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1, 2, 2)
    X, y = make_moons(n_samples=700, noise=0.05, random_state=2020)
    model = KMeans(n_clusters=2)
    model.fit(X)
    y_pred = model.predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, alpha=0.7)
    plt.title("月牙形",fontsize=15)
    plt.xticks([])
    plt.yticks([])
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    test_moon_circle()
    test_circle()
    test_moon()
    test_moon_circle_cluster()
