from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris


def load_data():
    data = load_iris()
    x, y = data.data, data.target
    return x, y


def train(x, y):
    model = LogisticRegression(multi_class='ovr')
    model.fit(x,y)
    print("Accuracy: ", model.score(x, y))


if __name__ == '__main__':
    x, y = load_data()
    train(x, y)
