import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def make_data():
    np.random.seed(20)
    x = np.random.rand(100) * 30 + 50  # square
    noise = np.random.rand(100) * 50
    y = x * 8 - 127  # price
    y = y - noise
    return x, y


def main(x, y):
    model = LinearRegression()
    x = np.reshape(x, (-1, 1))
    model.fit(np.reshape(x, (-1, 1)), y)
    y_pre = model.predict(x)
    plt.scatter(x, y)
    print("参数w={},b={}".format(model.coef_,model.intercept_))
    print("面积50的房价为：",model.predict([[50]]))
    plt.plot(x, y_pre, c='r')
    plt.xlabel('Square', fontsize=15)
    plt.ylabel('Price', fontsize=15)
    plt.tight_layout()  # 调整子图间距
    plt.show()


if __name__ == '__main__':
    x, y = make_data()
    main(x, y)
