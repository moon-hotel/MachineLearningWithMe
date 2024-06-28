from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def load_data():
    data = load_digits()
    x, y = data.data, data.target
    x_train, x_test, y_train, y_test = \
        train_test_split(x, y, test_size=0.3,
                         shuffle=True, random_state=20)
    return x_train, x_test, y_train, y_test


def train(x_train, x_test, y_train, y_test):
    model = SVC(C=1.0, kernel='linear')
    model.fit(x_train, y_train)
    y_pre = model.predict(x_test)
    print(f"准确率为：{model.score(x_test, y_test)}")
    print(classification_report(y_test, y_pre))


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = load_data()
    train(x_train, x_test, y_train, y_test)
    # 准确率为：0.975925925925926
    #               precision    recall  f1-score   support
    #
    #            0       1.00      1.00      1.00        50
    #            1       0.96      0.98      0.97        56
    #            2       1.00      1.00      1.00        44
    #            3       1.00      0.95      0.98        63
    #            4       1.00      1.00      1.00        60
    #            5       0.94      0.98      0.96        51
    #            6       1.00      0.97      0.98        59
    #            7       0.96      0.98      0.97        53
    #            8       0.94      0.98      0.96        52
    #            9       0.94      0.92      0.93        52
    #
    #     accuracy                           0.98       540
    #    macro avg       0.98      0.98      0.98       540
    # weighted avg       0.98      0.98      0.98       540
