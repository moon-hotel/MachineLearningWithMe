from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def load_data():
    data = load_digits()
    x, y = data.data, data.target
    x_train, x_test, y_train, y_test = \
        train_test_split(x, y, test_size=0.3,
                         shuffle=True, random_state=20)
    return x_train, x_test, y_train, y_test


def train(x_train, x_test, y_train, y_test):
    clf = SVC(kernel='linear')
    clf.fit(x_train, y_train)
    y_pre = clf.predict(x_test)
    print(classification_report(y_test,y_pre))


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = load_data()
    train(x_train, x_test, y_train, y_test)
