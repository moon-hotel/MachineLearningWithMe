from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_data():
    data = load_iris()
    x, y = data.data, data.target
    x_train, x_test, y_train, y_test \
        = train_test_split(x, y, test_size=0.3,random_state=1)
    ss = StandardScaler()
    x_train = ss.fit_transform(x_train)
    x_test = ss.transform(x_test)
    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = load_data()
    model = RandomForestClassifier(n_estimators=2, max_features=3,
                                   random_state=2)
    model.fit(x_train, y_train)
    print(model.score(x_test, y_test))

    # Fitting 3 folds for each of 8 candidates, totalling 24 fits
    # [CV] END final_estimator__criterion=gini, knn__n_neighbors=3, logist__max_iter=200; total time=   0.1s
    # [CV] END final_estimator__criterion=gini, knn__n_neighbors=3, logist__max_iter=200; total time=   0.1s
    # [CV] END final_estimator__criterion=gini, knn__n_neighbors=3, logist__max_iter=200; total time=   0.1s
    # [CV] END final_estimator__criterion=gini, knn__n_neighbors=3, logist__max_iter=500; total time=   0.1s
    # [CV] END final_estimator__criterion=gini, knn__n_neighbors=3, logist__max_iter=500; total time=   0.1s
    # [CV] END final_estimator__criterion=gini, knn__n_neighbors=3, logist__max_iter=500; total time=   0.1s
    # [CV] END final_estimator__criterion=gini, knn__n_neighbors=4, logist__max_iter=200; total time=   0.1s
    # [CV] END final_estimator__criterion=gini, knn__n_neighbors=4, logist__max_iter=200; total time=   0.1s
    # [CV] END final_estimator__criterion=gini, knn__n_neighbors=4, logist__max_iter=200; total time=   0.1s
    # [CV] END final_estimator__criterion=gini, knn__n_neighbors=4, logist__max_iter=500; total time=   0.1s
    # [CV] END final_estimator__criterion=gini, knn__n_neighbors=4, logist__max_iter=500; total time=   0.1s
    # [CV] END final_estimator__criterion=gini, knn__n_neighbors=4, logist__max_iter=500; total time=   0.1s
    # [CV] END final_estimator__criterion=entropy, knn__n_neighbors=3, logist__max_iter=200; total time=   0.1s
    # [CV] END final_estimator__criterion=entropy, knn__n_neighbors=3, logist__max_iter=200; total time=   0.1s
    # [CV] END final_estimator__criterion=entropy, knn__n_neighbors=3, logist__max_iter=200; total time=   0.1s
    # [CV] END final_estimator__criterion=entropy, knn__n_neighbors=3, logist__max_iter=500; total time=   0.1s
    # [CV] END final_estimator__criterion=entropy, knn__n_neighbors=3, logist__max_iter=500; total time=   0.1s
    # [CV] END final_estimator__criterion=entropy, knn__n_neighbors=3, logist__max_iter=500; total time=   0.1s
    # [CV] END final_estimator__criterion=entropy, knn__n_neighbors=4, logist__max_iter=200; total time=   0.1s
    # [CV] END final_estimator__criterion=entropy, knn__n_neighbors=4, logist__max_iter=200; total time=   0.1s
    # [CV] END final_estimator__criterion=entropy, knn__n_neighbors=4, logist__max_iter=200; total time=   0.1s
    # [CV] END final_estimator__criterion=entropy, knn__n_neighbors=4, logist__max_iter=500; total time=   0.1s
    # [CV] END final_estimator__criterion=entropy, knn__n_neighbors=4, logist__max_iter=500; total time=   0.1s
    # [CV] END final_estimator__criterion=entropy, knn__n_neighbors=4, logist__max_iter=500; total time=   0.1s
    # 模型在测试集上的准确率为： 0.9777777777777777

