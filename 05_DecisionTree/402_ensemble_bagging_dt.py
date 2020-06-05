from sklearn.ensemble import BaggingClassifier,RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def load_data():
    data = load_iris()
    x, y = data.data, data.target
    x_train, x_test, y_train, y_test \
        = train_test_split(x, y, test_size=0.3)
    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = load_data()
    bagging = BaggingClassifier(n_estimators=5,
                                max_samples=50,
                                max_features=3,
                                bootstrap=True,
                                bootstrap_features=False)
    bagging.fit(x_train, y_train)
    print(bagging.score(x_test, y_test))

    rfc = RandomForestClassifier(n_estimators=5)
    rfc.fit(x_train,y_train)
    print(rfc.score(x_test,y_test))
