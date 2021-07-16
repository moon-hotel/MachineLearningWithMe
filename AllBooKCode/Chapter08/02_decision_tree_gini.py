from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree
import graphviz


def load_data():
    data = load_iris()
    X, y = data.data, data.target
    feature_names = data.feature_names
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test, feature_names


def train(X_train, X_test, y_train, y_test, feature_names):
    model = tree.DecisionTreeClassifier(criterion='gini', min_samples_leaf=5, random_state=30)
    model.fit(X_train, y_train)
    print("在测试集上的准确率为：", model.score(X_test, y_test))
    dot_data = tree.export_graphviz(model, out_file=None,
                                    feature_names=feature_names,
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render('iris')
    print("特征重要性为：", model.feature_importances_)


if __name__ == '__main__':
    X_train, X_test, y_train, y_test, feature_names = load_data()
    print("特征为性为：", feature_names)
    train(X_train, X_test, y_train, y_test, feature_names)
