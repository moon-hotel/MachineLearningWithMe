from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
import numpy as np


def MAE(y, y_pre):
    return np.mean(np.abs(y - y_pre))


def MSE(y, y_pre):
    return np.mean((y - y_pre) ** 2)


def RMSE(y, y_pre):
    return np.sqrt(MSE(y, y_pre))


def MAPE(y, y_pre):
    return np.mean(np.abs((y - y_pre) / y))


def R2(y, y_pre):
    u = np.sum((y - y_pre) ** 2)
    v = np.sum((y - np.mean(y_pre)) ** 2)
    return 1 - (u / v)


def load_data():
    data = load_boston()
    x = data.data
    y = data.target
    return x, y


def train(x, y):
    model = LinearRegression()
    model.fit(x, y)
    y_pre = model.predict(x)
    print("model score: ", model.score(x, y))
    print("MAE: ", MAE(y, y_pre))
    print("MSE: ", MSE(y, y_pre))
    print("MAPE: ", MAPE(y, y_pre))
    print("R^2: ", R2(y, y_pre))


if __name__ == '__main__':
    x, y = load_data()
    train(x, y)
