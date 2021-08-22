from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def load_data():
    data = load_iris()
    x, y = data.data, data.target
    x_train, x_test, y_train, y_test \
        = train_test_split(x, y, test_size=0.3,random_state=10)
    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = load_data()
    bagging = BaggingClassifier(KNeighborsClassifier(n_neighbors=3),
                                n_estimators=5,
                                max_samples=0.8,
                                max_features=3,
                                bootstrap=True,
                                bootstrap_features=False)
    bagging.fit(x_train, y_train)
    print(bagging.estimators_features_)
    print(bagging.estimators_samples_)
    print(bagging.score(x_train, y_train))
    print(bagging.score(x_test, y_test))
