"""
文件名: AllBooKCode/Chapter04/C20_digits_classification.py
作 者: @空字符
B 站: @月来客栈Moon https://space.bilibili.com/392219165
知 乎: @月来客栈 https://www.zhihu.com/people/the_lastest
油 管: @月来客栈
小红书: @月来客栈
公众号: @月来客栈
代码仓库: https://github.com/moon-hotel/MachineLearningWithMe
"""

from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np


def load_data():
    data = load_digits()
    x, y = data.data, data.target
    x_train, x_test, y_train, y_test = \
        train_test_split(x, y, test_size=0.3, random_state=20)
    ss = StandardScaler()
    x_train = ss.fit_transform(x_train)
    x_test = ss.transform(x_test)
    return x_train, x_test, y_train, y_test


def visualization(x):
    images = x.reshape(-1, 8, 8)  # reshape成一张图片的形状
    fig, ax = plt.subplots(3, 5)
    for i, axi in enumerate(ax.flat):
        image = images[i]
        # axi.imshow(image,cmap='gray')
        axi.imshow(image)
        axi.set(xticks=[], yticks=[])
    plt.tight_layout()
    plt.show()


def model_selection(X, y, k=5):
    learning_rates = [0.001, 0.03, 0.01, 0.03, 0.1, 0.3, 1, 3]
    penalties = [0, 0.01, 0.03, 0.1, 0.3, 1, 3]
    all_models = []
    model_id = 1
    for learning_rate in learning_rates:
        for penalty in penalties:
            print(f"正在训练模型 {model_id}: learning_rate = {learning_rate}, penalty = {penalty}")
            model = SGDClassifier(loss='log_loss', penalty='l2', learning_rate='constant',
                                  eta0=learning_rate, alpha=penalty)
            kf = KFold(n_splits=k, shuffle=True, random_state=10)
            model_score = []
            for train_index, dev_index in kf.split(X):
                X_train, X_dev = X[train_index], X[dev_index]
                y_train, y_dev = y[train_index], y[dev_index]
                model.fit(X_train, y_train)
                s = model.score(X_dev, y_dev)
                model_score.append(s)
            model_id += 1
            all_models.append([np.mean(model_score), learning_rate, penalty])
    print("最优模型: ", sorted(all_models, reverse=True, key=lambda x: x[0])[0])


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = load_data()
    print(x_train.shape)
    print(x_test.shape)

    # 先执行以下两行代码获得最优模型，然后注释掉
    # visualization(x_train)

    # model_selection(x_train, y_train)
    # 正在训练模型 1: learning_rate = 0.001, penalty = 0
    # 正在训练模型 2: learning_rate = 0.001, penalty = 0.01
    # 正在训练模型 3: learning_rate = 0.001, penalty = 0.03
    # 正在训练模型 4: learning_rate = 0.001, penalty = 0.1
    # 正在训练模型 5: learning_rate = 0.001, penalty = 0.3
    # 正在训练模型 6: learning_rate = 0.001, penalty = 1
    # 正在训练模型 7: learning_rate = 0.001, penalty = 3
    # .....
    # 最优模型:  [0.9586163283374439, 0.03, 0]

    # 在下面的模型参数填入上面模型选择结束后的最优参数：
    model = SGDClassifier(loss='log_loss', penalty='l2', learning_rate='constant', eta0=0.03, alpha=0.0)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(classification_report(y_test, y_pred))#