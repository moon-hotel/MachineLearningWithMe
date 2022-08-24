from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.model_selection import train_test_split


def load_data(scale=True):
    data = load_wine()
    x, y,feature_names = data.data, data.target,data.feature_names
    x_train, x_test, y_train, y_test = \
        train_test_split(x, y, test_size=0.3,
                         shuffle=True, random_state=20)
    if scale:
        ss = StandardScaler()
        x_train = ss.fit_transform(x_train)
        x_test = ss.transform(x_test)
    return x_train, x_test, y_train, y_test,feature_names


def train(x_train, x_test, y_train, y_test):
    model = DecisionTreeClassifier(criterion="entropy",max_depth=3)
    model.fit(x_train, y_train)
    with open("visualization.dot", "w") as f:
        f = export_graphviz(model, feature_names=feature_names, out_file=f)
    print("Accuracy: ", model.score(x_test, y_test))


if __name__ == '__main__':
    x_train, x_test, y_train, y_test,feature_names = load_data()
    train(x_train, x_test, y_train, y_test)
