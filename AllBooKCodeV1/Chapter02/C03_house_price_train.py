from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression


def load_data():
    data = fetch_california_housing()
    x = data.data
    y = data.target
    return x, y


def train(x, y):
    print('X的形状：', x.shape)
    model = LinearRegression()
    model.fit(x, y)
    print("权重为：", model.coef_)
    print("偏置为：", model.intercept_)
    print("第12个房屋的预测价格：", model.predict(x[12, :].reshape(1, -1)))
    print("第12个房屋的真实价格：", y[12])


if __name__ == '__main__':
    x, y = load_data()
    train(x, y)
