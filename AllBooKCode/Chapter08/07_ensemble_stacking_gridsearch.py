from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV


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
    dt = DecisionTreeClassifier()
    paras = {"logist__max_iter": [200, 500],  # 用定义基模型时的名称logist__来索引对应基模型中的参数
             "knn__n_neighbors": [3, 4],  # 用定义基模型时的名称knn__来索引对应基模型中的参数
             "final_estimator__criterion": ['gini', 'entropy']}  # 用StackingClassifier中的"final_estimator__参数来索引组合模型中的参数
    stacking = StackingClassifier(estimators=estimators,
                                  final_estimator=dt)
    gs = GridSearchCV(stacking, paras, verbose=2, cv=3)
    gs.fit(x_train, y_train)
    acc = gs.score(x_test, y_test)
    print("模型在测试集上的准确率为：", acc)
