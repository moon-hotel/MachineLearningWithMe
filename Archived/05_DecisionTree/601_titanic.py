import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def load_data():
    train = pd.read_csv('./data/titanic_train.csv')
    test = pd.read_csv('./data/test.csv')
    selected_features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    x_train = train[selected_features]
    y_train = train['Survived']
    x_test = test[selected_features]
    x_train['Age'].fillna(x_train['Age'].mean(), inplace=True)  # 以均值填充
    # print(x_train['Embarked'].value_counts())
    x_train['Embarked'].fillna('S', inplace=True)

    x_test['Age'].fillna(x_test['Age'].mean(), inplace=True)
    x_test['Fare'].fillna(x_test['Fare'].mean(), inplace=True)

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
    paras = {'n_estimators': np.arange(10, 100, 10), 'criterion': ['gini', 'entropy'], 'max_depth': np.arange(5, 50, 5)}
    gs = GridSearchCV(model, paras, cv=5, verbose=1, n_jobs=-1)
    gs.fit(x_train, y_train)
    y_pre = gs.predict(x_test)
    print('best score:', gs.best_score_)
    print('best parameters:', gs.best_params_)
    print(y_pre)


if __name__ == '__main__':
    random_forest()
    # x_train, y_train, x_test=load_data()
    # print(x_train.shape)
