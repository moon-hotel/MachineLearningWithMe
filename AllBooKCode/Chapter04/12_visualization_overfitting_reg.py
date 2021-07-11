import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def MSE(y, y_pre):
    return np.mean((y - y_pre) ** 2)


def make_data():
    np.random.seed(10)
    x_train = np.linspace(0, 2 * np.pi, 12)  # 训练样本
    x_test = np.linspace(0, 2 * np.pi, 1000)
    y_test = np.sin(x_test)
    y_train = np.random.uniform(-0.3, 0.3, 12) + np.sin(x_train)
    return x_test.reshape(-1, 1), y_test, x_train.reshape(-1, 1), y_train


def visualization(x_test, y_test, x_train, y_train):
    plt.plot(x_test, y_test, label='Real Data')
    plt.scatter(x_train, y_train, c='black', label='Train Data')
    plt.tight_layout()  # 调整子图间距
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.legend(fontsize=13)
    plt.show()


def polynomial_regression(x_train, y_train, x_test, y_test, degree=2):
    poly = PolynomialFeatures(include_bias=False, degree=degree)
    x_mul = poly.fit_transform(x_train)
    model = LinearRegression()

    model.fit(x_mul, y_train)

    x_mul = poly.transform(x_test)
    y_pre = model.predict(x_mul)
    r2 = model.score(x_mul, y_test)
    return y_pre, r2


def prediction(x_test, y_test, x_train, y_train):
    y_pre_1, r2_1 = polynomial_regression(x_train, y_train, x_test, y_test, degree=1)
    y_pre_5, r2_5 = polynomial_regression(x_train, y_train, x_test, y_test, degree=5)
    y_pre_10, r2_10 = polynomial_regression(x_train, y_train, x_test, y_test, degree=10)

    plt.scatter(x_train, y_train, c='black', label='Train Data')
    plt.plot(x_test, y_pre_1, linestyle='--', label=r'$degree = 1, R^2 = {}$'.format(round(r2_1, 2)))
    plt.plot(x_test, y_pre_5, label=r'$degree = 5, R^2 = {}$'.format(round(r2_5, 2)))
    plt.plot(x_test, y_pre_10, linestyle='dashdot', label=r'$degree = 10, R^2 = {}$'.format(round(r2_10, 2)))
    plt.tight_layout()  # 调整子图间距
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.legend(fontsize=13)
    plt.show()


def train(x_train, y_train):
    y_pre_1, score_1 = polynomial_regression(x_train, y_train, x_train, y_train, degree=1)
    y_pre_5, score_5 = polynomial_regression(x_train, y_train, x_train, y_train, degree=5)
    y_pre_10, score_10 = polynomial_regression(x_train, y_train, x_train, y_train, degree=10)

    plt.scatter(x_train, y_train, label='Train Data')
    plt.plot(x_train, y_pre_1, linestyle='--', label=r'$degree = 1, R^2 = {}$'.format(round(score_1, 2)))
    plt.plot(x_train, y_pre_5, label=r'$degree = 5, R^2 = {}$'.format(round(score_5, 2)))
    plt.plot(x_train, y_pre_10, linestyle='dashdot',
             label=r'$degree = 10, R^2 = {}$'.format(round(score_10, 2)))
    plt.tight_layout()  # 调整子图间距
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.legend(fontsize=13)
    plt.show()


if __name__ == '__main__':
    x_test, y_test, x_train, y_train = make_data()
    visualization(x_test, y_test, x_train, y_train)  # 可视化训练样本
    train(x_train, y_train)  # 可视化拟合后的曲线
    prediction(x_test, y_test, x_train, y_train)  # 预测新样本的输出
