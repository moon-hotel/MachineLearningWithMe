from sklearn.datasets import load_iris,load_breast_cancer
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt


def feature_scalling(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return (X - mean) / std


def load_data():
    data = load_iris()
    x, y = data.data, data.target.reshape(-1, 1)
    x = feature_scalling(x)
    return x, y


def accuracy(y, y_pre):
    return np.mean((y.flatten() == y_pre.flatten()) * 1)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def hypothesis(X, W, bias):
    z = np.matmul(X, W) + bias
    h_x = sigmoid(z)
    return h_x


def prediction(X, W, bias, thre=0.5):
    class_type = len(W)
    prob = []
    for c in range(class_type):
        w, b = W[c], bias[c]
        h_x = hypothesis(X, w, b)
        prob.append(h_x)
    prob = np.hstack(prob)
    y_pre = np.argmax(prob, axis=1)
    return y_pre


def cost_function(X, y, W, bias):
    m, n = X.shape
    h_x = hypothesis(X, W, bias)
    cost = np.sum(y * np.log(h_x) + (1 - y) * np.log(1 - h_x))
    return -cost / m


def gradient_descent(X, y, W, bias, alpha):
    m, n = X.shape
    h_x = hypothesis(X, W, bias)
    grad_w = (1 / m) * np.matmul(X.T, (h_x - y))  # [n,m] @ [m,1]
    grad_b = (1 / m) * np.sum(h_x - y)
    W = W - alpha * grad_w  # 梯度下降
    bias = bias - alpha * grad_b
    return W, bias


def train_binary(X, y, iter=200):
    m, n = X.shape  # 506,13
    # W = np.random.uniform(-0.4, 0.4, n).reshape(n, 1)#0.946
    W = np.random.randn(n, 1)#0.953
    b = 0.3
    alpha = 0.5
    costs = []
    for i in range(iter):
        J = cost_function(X, y, W, b)
        costs.append(J)
        W, b = gradient_descent(X, y, W, b, alpha)
    return costs, W, b


def train(x, y, iter=1000):
    class_type = np.unique(y)
    costs = []
    W, b = [], []
    for c in class_type:
        label = (y == c) * 1
        tmp = train_binary(x, label, iter=iter)
        costs.append(tmp[0])
        W.append(tmp[1])
        b.append(tmp[2])
    costs = np.vstack(costs)
    costs = np.sum(costs, axis=0)
    y_pre = prediction(x, W, b)
    print(classification_report(y, y_pre))
    print('Accuracy by impleme: ', accuracy(y, y_pre))
    return costs


def train_by_sklearn(x, y):
    model = LogisticRegression()
    model.fit(x, y)
    print("Accuracy by sklearn: ", model.score(x, y))


if __name__ == '__main__':
    x, y = load_data()
    costs = train(x, y)
    train_by_sklearn(x, y)
    plt.plot(range(len(costs)), costs, label='cost')
    plt.legend(fontsize=15)
    plt.tight_layout()  # 调整子图间距
    plt.show()
