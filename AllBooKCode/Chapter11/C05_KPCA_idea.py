import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt


def make_nonlinear_cla_data():
    num_points = 500
    x, y = make_circles(n_samples=num_points, factor=0.2, noise=0.1,
                        random_state=np.random.seed(10))
    x = x.reshape(-1, 2)
    return x, y.reshape(-1, 1)


def featuere_map(x):
    X, Y = x[:, 0], x[:, 1]
    Z = 1 / np.exp(X ** 2 + Y ** 2)
    X = np.abs(X) * 10
    Y = np.abs(Y) * 10
    return X, Y, Z


def visualization3D():
    fig = plt.figure(figsize=(6, 6))

    x_orig, y = make_nonlinear_cla_data()
    ax = fig.add_subplot(111, projection='3d')
    plt.title('Mapped Projection', fontsize=12)
    X, Y, Z = featuere_map(x_orig)
    ax.scatter(X, Y, Z, s=30, c=y)
    plt.tight_layout()
    plt.show()


def visualization():
    x_orig, y = make_nonlinear_cla_data()
    X, Y, Z = featuere_map(x_orig)
    x_map = np.hstack((np.reshape(X, (-1, 1)),
                       np.reshape(Y, (-1, 1)),
                       np.reshape(Z, (-1, 1))))
    pca = PCA(n_components=2)
    x = pca.fit_transform(x_map)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Projection with two components', fontsize=15)
    plt.scatter(x[:, 0], x[:, 1], c=y)
    plt.subplot(1, 2, 2)
    plt.title('Projection with one component', fontsize=15)
    plt.scatter(x[:, 0], [0] * len(x), c=y)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    visualization3D()
    visualization()
