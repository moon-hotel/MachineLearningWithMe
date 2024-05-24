from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris


def load_data():
    data = load_iris()
    x, y = data.data, data.target
    x_train, x_test, y_train, y_test \
        = train_test_split(x, y, test_size=0.3)
    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = load_data()
    estimators = [('logist', LogisticRegression(max_iter=500)),
                  ('knn', KNeighborsClassifier(n_neighbors=3))]
    stacking = StackingClassifier(estimators=estimators,
                                  final_estimator=DecisionTreeClassifier())
    stacking.fit(x_train, y_train)
    acc = stacking.score(x_test, y_test)
    print("模型在测试集上的准确率为：", acc)
