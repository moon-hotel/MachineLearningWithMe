from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split


def load_data(scale=False):
    data = load_digits()
    x, y = data.data, data.target
    x_train, x_test, y_train, y_test = \
        train_test_split(x, y, test_size=0.3, random_state=10)
    if scale:
        ss = StandardScaler()
        x_train = ss.fit_transform(x_train)
        x_test = ss.transform(x_test)
    return x_train, x_test, y_train, y_test


def model_selection(x_train, y_train):
    model = KNeighborsClassifier()
    paras = {'n_neighbors': [5, 6, 7, 8, 9, 10], 'p': [1, 2]}
    gs = GridSearchCV(model, paras, verbose=2, cv=5)
    gs.fit(x_train, y_train)
    print('最佳模型:', gs.best_params_, gs.best_score_)


def train(x_train, x_test, y_train, y_test):
    model = KNeighborsClassifier(5, p=2)
    model.fit(x_train, y_train)
    print("Accuracy: ", model.score(x_test, y_test))


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = load_data()
    # model_selection(x_train, y_train)
    train(x_train, x_test, y_train, y_test)
