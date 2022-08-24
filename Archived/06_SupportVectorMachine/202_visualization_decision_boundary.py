import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs


def decision_boundary():
    # 构造数据
    X, y = make_blobs(n_samples=80, centers=2, cluster_std=1.2, random_state=6)
    xlim = [np.min(X[:, 0]), np.max(X[:, 0])]
    ylim = [np.min(X[:, 1]), np.max(X[:, 1])]

    xx = np.linspace(xlim[0], xlim[1], 500)
    yy = np.linspace(ylim[0], ylim[1], 500)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T

    plt.figure(figsize=(7, 7), dpi=80)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Paired)
    clf = svm.SVC(kernel='linear', C=1000)
    clf.fit(X, y)
    Z4 = clf.decision_function(xy).reshape(XX.shape)
    plt.contour(XX, YY, Z4, colors='green', levels=[-1, 0, 1], alpha=0.8, linestyles=['--', '-', '--'])
    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
                linewidth=1, facecolors='none', edgecolors='k')
    plt.annotate(r'$\;$', xy=(7.15,-5.11), fontsize=16,
                 xytext=(6.78,-6.5), arrowprops=dict(arrowstyle="<->", connectionstyle="arc3"))
    plt.annotate(r'$d$', xy=(7.15, -5.81), fontsize=18, c='r')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    decision_boundary()
