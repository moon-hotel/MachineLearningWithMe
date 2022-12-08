"""
文件名: C05_bag_of_word_cla.py
创建时间:
作者: @空字符
公众号: @月来客栈
知乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import joblib
import os
import sys

sys.path.append('../')  # 把上一层整个目录加入到系统环境变量中
from utils import load_cut_spam  # 在utils里的dataset.py模块中


def get_dataset():
    x, y = load_cut_spam()
    X_train, X_test, y_train, y_test = \
        train_test_split(x, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


def preprocessing(x, top_k_words=1000, MODEL_NAME='count_vec.pkl'):
    count_vec = load_model(MODEL_NAME=MODEL_NAME)
    if not count_vec:  # 模型不存在
        count_vec = CountVectorizer(max_features=top_k_words)
        ## 考虑词频的词袋模型
        count_vec.fit(x)  # 重新训练
        # print(len(count_vec.vocabulary_)) # 输出词表长度
        save_model(count_vec, MODEL_NAME='count_vec.pkl')
    x = count_vec.transform(x)
    return x


def save_model(model, dir='MODEL', MODEL_NAME='model.pkl'):
    if not os.path.exists(dir):
        os.mkdir(dir)
    joblib.dump(model, os.path.join(dir, MODEL_NAME))


def load_model(dir='MODEL', MODEL_NAME='model.pkl'):
    path = os.path.join(dir, MODEL_NAME)
    if os.path.exists(path):
        print(f"载入已有模型: {path}")
        model = joblib.load(path)
        return model
    return False


def train(X_train, y_train):
    X_train = preprocessing(X_train)
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    save_model(model, MODEL_NAME='KNN.pkl')
    y_pred = model.predict(X_train)
    print("模型在训练集上的表现结果：")
    print(classification_report(y_train, y_pred))


def predict(X, MODEL_NAME='KNN.pkl'):
    X_test = preprocessing(X)
    model = load_model(MODEL_NAME=MODEL_NAME)
    if not model:
        raise FileNotFoundError(f"模型 {MODEL_NAME} 不存在，请先执行 train() 训练模型！")
    y_pred = model.predict(X_test)
    return y_pred


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_dataset()
    # train(X_train, y_train)
    y_pred = predict(X_test)
    print("模型在测试集上的表现结果：")
    print(classification_report(y_test, y_pred))
