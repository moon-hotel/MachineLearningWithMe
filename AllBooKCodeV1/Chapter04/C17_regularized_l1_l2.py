from sklearn.datasets import fetch_california_housing
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler


def regression_penalty():
    x, y = fetch_california_housing(return_X_y=True)
    x_train, x_test, y_train, y_test = \
        train_test_split(x, y, test_size=0.3, random_state=2020)
    ss = StandardScaler()
    x_train = ss.fit_transform(x_train)
    x_test = ss.transform(x_test)
    model = SGDRegressor(loss="squared_error",l1_ratio=0.4, penalty='elasticnet', alpha=0.001)
    model.fit(x_train, y_train)
    print(model.predict(x_test)[:5])
    print(y_test[:5])


def classification_penalty():
    x, y = load_breast_cancer(return_X_y=True)
    x_train, x_test, y_train, y_test = \
        train_test_split(x, y, test_size=0.3, random_state=2020)
    ss = StandardScaler()
    x_train = ss.fit_transform(x_train)
    x_test = ss.transform(x_test)
    model = SGDClassifier(loss="log_loss",l1_ratio=0.4, penalty='elasticnet', alpha=0.001)
    model.fit(x_train, y_train)
    print(model.score(x_test, y_test))
    print(model.predict(x_test)[:5])
    print(y_test[:5])


if __name__ == '__main__':
    # regression_penalty()
    classification_penalty()
