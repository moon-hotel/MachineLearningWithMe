import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.linear_model import LinearRegression


def make_data():
    num_points = 200
    centers = [[1, 1], [2, 2]]  # 指定中心
    x, y = make_blobs(n_samples=num_points, centers=centers,
                      cluster_std=0.2, random_state=np.random.seed(10))
    index_pos, index_neg = (y == 1), (y == 0)
    x_pos, x_neg = x[index_pos], x[index_neg]
    plt.scatter(x_pos[:, 0], x_pos[:, 1], marker='o', label='positive')
    plt.scatter(x_neg[:, 0], x_neg[:, 1], marker='s', label='negative')
    plt.legend(fontsize=15)
    return x, y


def decision_boundary(x, y):
    ###########  模型求解并预测
    x, y = x[:, 0].reshape(-1, 1), x[:, 1]
    model = LinearRegression()
    model.fit(x, y)
    y_pre = model.predict(x)
    plt.plot(x, y_pre, c='r')
    plt.tight_layout()  # 调整子图间距
    plt.show()


if __name__ == '__main__':
    x, y = make_data()
    decision_boundary(x, y)
