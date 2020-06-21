import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_blobs


def decision_boundary():
    # 构造数据
    X, y = make_blobs(n_samples=80, centers=2, cluster_std=1.2, random_state=6)
    xlim = [np.min(X[:, 0]), np.max(X[:, 0])]
    ylim = [np.min(X[:, 1]), np.max(X[:, 1])]
    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 500)
    yy = np.linspace(ylim[0], ylim[1], 500)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T

    plt.figure(figsize=(12, 4), dpi=80)
    plt.subplot(1, 3, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Paired)
    plt.xlabel('(a)', fontsize=15)

    plt.subplot(1, 3, 2)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Paired)
    clf = LogisticRegression()
    clf.fit(X, y)
    Z1 = np.matmul(xy, clf.coef_.T-1.8) + clf.intercept_
    Z1 =Z1.reshape(XX.shape)
    plt.contour(XX, YY, Z1, colors='red', levels=[-4.5, 0, 1.8], alpha=0.8, linestyles=['--', '-', '--'])
    plt.xlabel('(b)', fontsize=15)

    plt.subplot(1, 3, 3)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Paired)
    clf = svm.SVC(kernel='linear', C=1000)
    clf.fit(X, y)
    Z4 = clf.decision_function(xy).reshape(XX.shape)
    plt.contour(XX, YY, Z4, colors='green', levels=[-1, 0, 1], alpha=0.8, linestyles=['--', '-', '--'])
    plt.xlabel('(c)', fontsize=15)
    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
                linewidth=1, facecolors='none', edgecolors='k')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    decision_boundary()
