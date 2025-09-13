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
    model.fit(x, y)
    y_pre = model.predict(x)
    plt.rcParams['ytick.direction'] = 'in'  # 刻度向内
    plt.rcParams['xtick.direction'] = 'in'  # 刻度向内
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.scatter(x, y, c='black')
    print("参数w={},b={}".format(model.coef_,model.intercept_))
    print("面积50的房价为：",model.predict([[50]]))
    plt.plot(x, y_pre, c='black')
    plt.xlabel('面积', fontsize=15)
    plt.ylabel('房价', fontsize=15)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.tight_layout()  # 调整子图间距
    plt.show()


if __name__ == '__main__':
    x, y = make_data()
    main(x, y)
