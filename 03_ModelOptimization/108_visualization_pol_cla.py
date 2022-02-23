import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures


def standarlization(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return (X - mean) / std


def make_nonlinear_cla_data():
    num_points = 200
    x, y = make_circles(num_points, factor=0.5, noise=0.06, random_state=np.random.seed(10))
    x = x.reshape(-1, 2)
    x = standarlization(x)
    return x, y.reshape(-1, 1)


def decision_boundary(x, y, pol):
    model = LogisticRegression()
    model.fit(x, y)
    print("Accuracy:", model.score(x, y))
    x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5
    y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5
    h = .01  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    x_new = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])
    Z = model.predict(pol.transform(x_new))

    Z = Z.reshape(xx.shape)
    plt.figure(1, figsize=(6, 5))
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired, shading='auto')
    plt.scatter(x[:, 0], x[:, 1], c=y, alpha=0.83)  # alpha 控制透明度
    plt.show()


if __name__ == '__main__':
    x, y = make_nonlinear_cla_data()
    pol = PolynomialFeatures(degree=2, include_bias=False)
    x_pol = pol.fit_transform(x)

    decision_boundary(x_pol, y, pol)
