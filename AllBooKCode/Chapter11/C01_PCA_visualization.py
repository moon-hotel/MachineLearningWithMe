import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


def make_data_visualization():
    rng = np.random.RandomState(0)
    n_samples = 300
    cov = [[3, 3], [3, 4]]
    X = rng.multivariate_normal(mean=[0, 0], cov=cov, size=n_samples)  # 构造一个二元正态分布数据集
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], alpha=0.6, label="samples")
    plt.gca().set(aspect="equal")  # x,y轴按等比例进行展示

    plt.subplot(1, 2, 2)
    pca = PCA(n_components=2).fit(X)
    plt.scatter(X[:, 0], X[:, 1], alpha=0.5, label="samples")
    for i, (comp, var) in enumerate(zip(pca.components_, pca.explained_variance_)):
        comp = comp * var  # scale component by its variance explanation power
        plt.arrow(0, 0, comp[0], comp[1], head_width=0.1,
                  head_length=0.1, linewidth=2, color=f"C{i + 2}", label=f"Component {i}")
    plt.gca().set(aspect="equal")  # x,y轴按等比例进行展示
    plt.legend()
    plt.tight_layout()
    plt.show()


def eigenvalue_vs_singular():
    rng = np.random.RandomState(0)
    n_samples = 500
    cov = [[3, 3], [3, 4]]
    X = rng.multivariate_normal(mean=[0, 0], cov=cov, size=n_samples)  # 构造一个二元正态分布数据集

    plt.figure(figsize=(6, 4))
    plt.subplot(1, 2, 1)
    pca = PCA(n_components=2).fit(X)
    plt.scatter(X[:, 0], X[:, 1], alpha=0.5, label="samples")
    for i, (comp, var) in enumerate(zip(pca.components_, pca.explained_variance_)):
        comp = comp * var  # scale component by its variance explanation power
        plt.arrow(0, 0, comp[0], comp[1], head_width=0.1,
                  head_length=0.1, linewidth=2, color=f"C{i + 2}", label=f"Component {i}")
    plt.gca().set(aspect="equal", title="2-dimensional dataset with principal components",
                  xlabel="first feature", ylabel="second feature", )  # x,y轴按等比例进行展示
    plt.legend()

    plt.subplot(1, 2, 2)
    w, v = np.linalg.eig(np.matmul(X.T, X) / len(X))
    # v[:,i] 是特征值w[i]所对应的特征向量
    plt.scatter(X[:, 0], X[:, 1], alpha=0.5, label="samples")
    for i, (comp, var) in enumerate(zip(v.T, w)):
        comp = comp * var  # scale component by its variance explanation power
        plt.arrow(0, 0, comp[0], comp[1], head_width=0.1,
                  head_length=0.1, linewidth=2, color=f"C{i + 2}", label=f"Component {i}")
    plt.gca().set(aspect="equal", title="2-dimensional dataset with principal components",
                  xlabel="first feature", ylabel="second feature", )  # x,y轴按等比例进行展示
    plt.legend()
    plt.tight_layout()
    plt.show()


def visualization_3d():
    def pdf(x):
        return 0.5 * (stats.norm(scale=0.25 / e).pdf(x) + stats.norm(scale=4 / e).pdf(x))

    from scipy import stats
    e = np.exp(1)
    np.random.seed(4)

    y = np.random.normal(scale=0.5, size=(500))
    x = np.random.normal(scale=0.5, size=(500))
    z = np.random.normal(scale=0.1, size=len(x))

    density = pdf(x) * pdf(y)
    pdf_z = pdf(5 * z)

    density *= pdf_z

    a = x + y
    b = 2 * y
    c = a - b + z

    norm = np.sqrt(a.var() + b.var())
    a /= norm
    b /= norm

    X = np.c_[a, b, c]
    fig = plt.figure()
    ax = Axes3D(fig)
    pca = PCA(n_components=3).fit(X)
    ax.scatter3D(X[:, 0], X[:, 1], X[:, 2], alpha=0.5, label="samples")
    for i, (comp, var) in enumerate(zip(pca.components_, pca.explained_variance_)):
        comp = comp * 2.3  # scale component by its variance explanation power
        ax.quiver(0, 0, 0, comp[0], comp[1], comp[2], color=f"C{i + 2}",
                  linewidth=2, label=f"Component {i}", arrow_length_ratio=.2)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_zlim(-2.5, 2.5)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    make_data_visualization()
    visualization_3d()
    eigenvalue_vs_singular()
