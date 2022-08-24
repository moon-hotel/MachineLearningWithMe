from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


def visualization():
    centers = [[2.2, 1], [3.8, 1], [3, 2.8]]  # 指定簇中心
    x, y = make_blobs(n_samples=800, centers=centers, cluster_std=0.3, random_state=200)
    plt.scatter(x[:, 0], x[:, 1], c=y, s=40, cmap=plt.cm.Paired)
    plt.xticks([])
    plt.yticks([])
    plt.show()


if __name__ == '__main__':
    visualization()
