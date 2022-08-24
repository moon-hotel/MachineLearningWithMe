from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import numpy as np


def load_data():
    data = load_digits()
    x, y = data.data, data.target
    x_train, x_test, y_train, y_test = \
        train_test_split(x, y, test_size=0.3,
                         shuffle=True, random_state=20)
    return x_train, x_test, y_train, y_test


def model_selection(x_train, y_train):
    model = SVC()
    paras = {'C': np.arange(1, 100, 5), 'kernel': ['rbf', 'linear', 'poly'], 'degree': np.arange(1, 20, 2)}
    gs = GridSearchCV(model, paras, cv=5, verbose=1, n_jobs=-1)
    gs.fit(x_train, y_train)
    print('best score:', gs.best_score_)
    print('best parameters:', gs.best_params_)


def train(x_train, x_test, y_train, y_test):
    model = SVC(C=6, kernel='rbf')
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    print("accuracy: ", score)


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = load_data()
    model_selection(x_train, y_train)
    train(x_train, x_test, y_train, y_test)
