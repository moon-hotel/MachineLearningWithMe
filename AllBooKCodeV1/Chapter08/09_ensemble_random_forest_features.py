from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import graphviz
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def load_data():
    data = load_iris()
    x, y = data.data, data.target
    feature_names = data.feature_names
    x_train, x_test, y_train, y_test \
        = train_test_split(x, y, test_size=0.3, random_state=1)
    return x_train, x_test, y_train, y_test, feature_names


if __name__ == '__main__':
    x_train, x_test, y_train, y_test, feature_names = load_data()
    print(x_train.shape)
    model = RandomForestClassifier(n_estimators=2,
                                   random_state=2,
                                   max_features=3,
                                   )
    model.fit(x_train, y_train)

    print("所有的决策树模型：", model.estimators_)  # 输出所有的决策树
    imps = []
    print(model.n_outputs_)
    for i in range(2):
        dot_data = tree.export_graphviz(model.estimators_[i], out_file=None,
                                        feature_names=feature_names,
                                        # filled=True, rounded=True,
                                        special_characters=True)
        imp = model.estimators_[i].tree_.compute_feature_importances(normalize=False)

        # print(model.estimators_[i].tree_.value)
        # print(model.estimators_[i].tree_.impurity)
        imps.append(imp)
        graph = graphviz.Source(dot_data)
        graph.render(f"iris{i}")
    # print(model.score(x_test, y_test))
    import numpy as np

    print("特征名称：", feature_names)
    print("feature_importances_：", model.feature_importances_)
    a = np.vstack(imps)
    print("每个决策树各自的特征重要性(未标准化)：\n", a)
    print("每个决策树各自的特征重要性(标准化后)：\n", a / a.sum(axis=1, keepdims=True))
    print("随机森林计算得到的特征重要性：")
    imp_s = a / a.sum(axis=1, keepdims=True)
    print(imp_s.mean(axis=0))

    # (105, 4)
    # 所有的决策树模型： [DecisionTreeClassifier(max_features=3, random_state=1872583848), DecisionTreeClassifier(max_features=3, random_state=794921487)]
    # 1
    # 特征名称： ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    # feature_importances_： [0.         0.         0.72939266 0.27060734]
    # 每个决策树各自的特征重要性(未标准化)：
    #  [[0.         0.         0.33823129 0.31900227]
    #  [0.         0.         0.6189899  0.03661101]]
    # 每个决策树各自的特征重要性(标准化后)：
    #  [[0.         0.         0.51462876 0.48537124]
    #  [0.         0.         0.94415656 0.05584344]]
    # 随机森林计算得到的特征重要性：
    # [0.         0.         0.72939266 0.27060734]
