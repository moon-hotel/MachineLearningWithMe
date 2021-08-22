from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


def load_data():
    data = load_iris()
    x, y = data.data, data.target
    x_train, x_test, y_train, y_test \
        = train_test_split(x, y, test_size=0.3, random_state=10)
    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = load_data()
    dt = DecisionTreeClassifier()
    paras = {"base_estimator__criterion": ['gini', 'entropy'],
             "base_estimator__max_depth": [1, 2],#  base_estimator__ 来索引基模型中的各个参数
             "n_estimators": [20, 30, 50, 100]}

    ada = AdaBoostClassifier(base_estimator=dt)
    gs = GridSearchCV(ada, paras, verbose=2, cv=3)
    gs.fit(x_train, y_train)
    print('最佳模型:', gs.best_params_, '准确率：', gs.best_score_)
    print("模型在测试集上的准确率为：", gs.score(x_test, y_test))
