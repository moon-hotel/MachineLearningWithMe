from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np


def load_data(scale=True):
    data = load_digits()
    x, y = data.data, data.target
    x_train, x_test, y_train, y_test = \
        train_test_split(x, y, test_size=0.3,
                         shuffle=True, random_state=20)
    if scale:
        ss = StandardScaler()
        x_train = ss.fit_transform(x_train)
        x_test = ss.transform(x_test)
    return x_train, x_test, y_train, y_test


def visualization(x):
    images = x.reshape(-1, 8, 8)  # reshape成一张图片的形状
    fig, ax = plt.subplots(3, 5)
    for i, axi in enumerate(ax.flat):
        image = images[i]
        axi.imshow(image)
        axi.set(xticks=[], yticks=[])
    plt.tight_layout()
    plt.show()


def model_selection(X, y, k=5):
    alphas = [1e-5,3e-5,1e-4,3e-4,1e-3,3e-3, 0.01, 0.03, 0.1, 0.3, 1, 3]
    all_models = []
    for al in alphas:
        model = SGDClassifier(loss='log',
                              penalty='l2', alpha=al)
        kf = KFold(n_splits=k, shuffle=True, random_state=10)
        model_score = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model.fit(X_train, y_train)
            s = model.score(X_test, y_test)
            model_score.append(s)
        all_models.append([np.mean(model_score), al])
    print("The best model: ", sorted(all_models, reverse=True)[0])


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = load_data(scale=True)
    print(x_train.shape)
    print(x_test.shape)
    # visualization(x_train)
    # model_selection(x_train, y_train)  # The best model:  [0.9490956807689876, 0.0001]

    model = SGDClassifier(loss='log', penalty='l2', alpha=0.00003)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(classification_report(y_test, y_pred))
