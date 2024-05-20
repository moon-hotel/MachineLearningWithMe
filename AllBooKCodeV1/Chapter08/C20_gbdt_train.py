from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    x, y = load_iris(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    model = GradientBoostingClassifier(learning_rate=0.2, n_estimators=50)
    model.fit(x_train, y_train)
    print(model.score(x_test, y_test))
