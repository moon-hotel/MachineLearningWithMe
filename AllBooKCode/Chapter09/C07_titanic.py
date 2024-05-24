import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def load_data():
    # 读取原始数据
    train = pd.read_csv('data/train.csv', sep=',')
    test = pd.read_csv('data/test.csv')

    # 进行特征选择
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    x_train = train[features].copy()
    y_train = train['Survived']
    x_test = test[features].copy()

    # 缺失值填充
    x_train['Age'] = x_train['Age'].fillna(x_train['Age'].mean())  # 以均值填充
    print(x_train['Embarked'].value_counts())  # 统计Embarked中，各个取值的出现次数
    x_train['Embarked'] = x_train['Embarked'].fillna('S')
    x_test['Age'] = x_test['Age'].fillna(x_train['Age'].mean())
    x_test['Fare'] = x_test['Fare'].fillna(x_train['Fare'].mean())

    # 特征转换
    x_train.loc[x_train['Sex'] == 'male', 'Sex'] = 0
    x_train.loc[x_train['Sex'] == 'female', 'Sex'] = 1
    x_train.loc[x_train['Embarked'] == 'S', 'Embarked'] = 0
    x_train.loc[x_train['Embarked'] == 'C', 'Embarked'] = 1
    x_train.loc[x_train['Embarked'] == 'Q', 'Embarked'] = 2

    x_test.loc[x_test['Sex'] == 'male', 'Sex'] = 0
    x_test.loc[x_test['Sex'] == 'female', 'Sex'] = 1
    x_test.loc[x_test['Embarked'] == 'S', 'Embarked'] = 0
    x_test.loc[x_test['Embarked'] == 'C', 'Embarked'] = 1
    x_test.loc[x_test['Embarked'] == 'Q', 'Embarked'] = 2
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    return x_train, y_train, x_test


def random_forest():
    x_train, y_train, x_test = load_data()
    model = RandomForestClassifier()
    paras = {'n_estimators': np.arange(10, 100, 10), 'criterion': ['gini', 'entropy'], 'max_depth': np.arange(5, 20, 5)}
    gs = GridSearchCV(estimator=model, param_grid=paras, cv=3, verbose=2, n_jobs=1)
    gs.fit(x_train, y_train)
    y_pre = gs.predict(x_test)
    print('best score:', gs.best_score_)
    print('best parameters:', gs.best_params_)
    print(y_pre)


if __name__ == '__main__':
    random_forest()

    # Embarked
    # S    644
    # C    168
    # Q     77
    # Name: count, dtype: int64
    # Fitting 3 folds for each of 54 candidates, totalling 162 fits
    # [CV] END .......criterion=gini, max_depth=5, n_estimators=10; total time=   0.0s
    # [CV] END .......criterion=gini, max_depth=5, n_estimators=10; total time=   0.0s
    # [CV] END .......criterion=gini, max_depth=5, n_estimators=10; total time=   0.0s
    # [CV] END .......criterion=gini, max_depth=5, n_estimators=20; total time=   0.0s
    # [CV] END .......criterion=gini, max_depth=5, n_estimators=20; total time=   0.0s
    # [CV] END .......criterion=gini, max_depth=5, n_estimators=20; total time=   0.0s
    # .....
    # [CV] END ...criterion=entropy, max_depth=15, n_estimators=90; total time=   0.1s
    # best score: 0.8338945005611672
    # best parameters: {'criterion': 'entropy', 'max_depth': 10, 'n_estimators': 30}
    # [0 0 0 0 0 0 1 0 1 0 0 0 1 0 1 1 0 0 0 0 1 1 1 1 1 0 1 0 0 0 0 0 0 0 0 0 0
    #  0 0 1 0 1 0 1 1 0 0 0 1 0 0 0 1 1 0 0 0 0 0 1 0 0 0 1 1 1 1 0 0 1 1 0 0 0
    #  1 0 0 1 0 1 1 0 0 0 0 0 1 0 1 1 0 0 1 0 0 0 1 0 1 0 1 0 0 0 0 0 0 0 0 0 0
    #  1 1 1 1 0 0 1 0 1 1 0 1 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 0 0 0
    #  1 0 1 0 0 1 0 0 1 1 1 1 1 1 1 0 0 0 0 0 1 0 0 0 0 0 0 1 1 1 1 1 0 0 1 0 1
    #  0 1 0 0 0 0 0 1 0 1 0 1 0 0 0 1 1 0 1 0 0 0 0 1 0 0 0 0 1 1 0 1 0 1 0 1 0
    #  1 0 1 1 0 1 0 0 0 1 0 0 1 0 1 0 1 1 1 1 0 0 0 0 1 0 1 0 1 0 0 0 0 0 0 0 1
    #  0 0 0 1 1 0 0 0 0 0 0 0 0 1 1 0 1 0 0 0 0 0 1 1 0 1 0 0 0 0 0 0 0 0 0 0 0
    #  1 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 1 1 0 1 0 0 0 1 0 0
    #  1 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 1 1 0 0 0 1 0 1 0 0 1 0 1 1 0 1 0 0 0 1 0
    #  0 1 0 0 1 1 0 0 0 0 0 0 1 0 0 1 0 0 0 0 0 1 1 0 0 1 0 1 0 0 1 0 1 0 0 0 0
    #  0 1 1 1 1 0 0 1 0 0 1]
