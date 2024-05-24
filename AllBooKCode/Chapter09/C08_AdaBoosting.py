from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def load_data():
    data = load_iris()
    x, y = data.data, data.target
    x_train, x_test, y_train, y_test \
        = train_test_split(x, y, test_size=0.3, random_state=10)
    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = load_data()
    dt = DecisionTreeClassifier(criterion='gini', max_features=4, max_depth=1)
    model = AdaBoostClassifier(estimator=dt, n_estimators=100)
    model.fit(x_train, y_train)
    print("模型在测试集上的准确率为：", model.score(x_test, y_test))
