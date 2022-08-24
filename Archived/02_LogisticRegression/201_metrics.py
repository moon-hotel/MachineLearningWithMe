from sklearn.metrics import classification_report
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression


def load_data():
    data = load_breast_cancer()
    # data = load_iris()
    x = data.data
    y = data.target
    label_names = list(data.target_names)
    return x, y, label_names


def train(x, y, label_names):
    model = LogisticRegression()
    model.fit(x, y)
    print(x.shape)
    accuracy = model.score(x, y)
    y_pre = model.predict(x)
    print("Accuracy: ", accuracy)
    print(classification_report(y, y_pre, target_names=label_names))


if __name__ == '__main__':
    x, y, label_names = load_data()
    train(x, y, label_names)
