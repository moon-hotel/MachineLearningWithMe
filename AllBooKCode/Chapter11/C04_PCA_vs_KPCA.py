import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import PCA
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt


def make_nonlinear_cla_data():
    num_points = 500
    x, y = make_circles(n_samples=num_points, factor=0.2, noise=0.1,
                        random_state=np.random.seed(10))
    x = x.reshape(-1, 2)
    return x, y.reshape(-1, 1)


def visualization():
    plt.figure(figsize=(15, 5), dpi=80)
    plt.subplot(1, 3, 1)
    plt.title('Original Projection')
    x_orig, y = make_nonlinear_cla_data()
    plt.scatter(x_orig[:, 0], x_orig[:, 1], c=y)

    plt.subplot(1, 3, 2)
    plt.title('Projection via PCA')
    pca = PCA(n_components=1)
    x = pca.fit_transform(x_orig)
    plt.scatter(range(len(x)), x[:, 0], c=y)

    plt.subplot(1, 3, 3)
    plt.title('Projection via KernelPCA')
    pca = KernelPCA(n_components=1, kernel='rbf', gamma=10)
    x = pca.fit_transform(x_orig)
    plt.scatter(x[:, 0], [0] * len(x), c=y)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    visualization()
