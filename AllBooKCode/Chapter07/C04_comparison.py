"""
文件名: AllBooKCode/Chapter07/C04_comparison.py
创建时间: 2022年
作者: @空字符
公众号: @月来客栈
知乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""
from C01_naive_bayes_category import MyCategoricalNB
from C02_naive_bayes_multinomial import MyMultinomialNB
from C03_naive_bayes_gaussian import MyGaussianNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
import sys

sys.path.append('../')
from utils import load_cut_spam
from utils import VectWithoutFrequency


def make_dataset(feature_type='category', top_k_words=1000):
    x_text, y = load_cut_spam()
    if feature_type == "category":
        vectorizer = VectWithoutFrequency(top_k_words=top_k_words)
    elif feature_type == "counts":
        vectorizer = CountVectorizer(max_features=top_k_words)
    elif feature_type == "tfidf":
        vectorizer = TfidfVectorizer(max_features=top_k_words)
    else:
        raise ValueError(f"不存在特征处理类型{feature_type}")
    x_train, x_test, y_train, y_test = train_test_split(
        x_text, y, test_size=0.3, random_state=2022)
    x_train = vectorizer.fit_transform(x_train)
    x_test = vectorizer.transform(x_test)
    if not isinstance(x_train, np.ndarray):
        x_train = x_train.toarray()
        x_test = x_test.toarray()
    return x_train, x_test, y_train, y_test


def comp_funcs(x_train, x_test, y_train, y_test, model_type):
    if model_type == "MyCategoricalNB":
        model = MyCategoricalNB()
    elif model_type == "MyMultinomialNB":
        model = MyMultinomialNB()
    else:
        model = MyGaussianNB()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(f"{model_type}模型在测试集上的准确率为:{accuracy_score(y_test, y_pred)}")


def comparison(feature_type="category", top_k_words=1000, model_type="MyCategoricalNB"):
    print(f"{feature_type}特征表示方法，", end='')
    x_train, x_test, y_train, y_test = make_dataset(feature_type, top_k_words)
    comp_funcs(x_train, x_test, y_train, y_test, model_type)


if __name__ == '__main__':
    feature_types = ["category", "counts", "tfidf"]
    model_types = ["MyCategoricalNB", "MyMultinomialNB", "MyGaussianNB"]
    top_k_words = 6000
    comparison(feature_types[0], top_k_words, model_types[0])
    comparison(feature_types[0], top_k_words, model_types[1])
    comparison(feature_types[0], top_k_words, model_types[2])
    comparison(feature_types[1], top_k_words, model_types[1])
    comparison(feature_types[1], top_k_words, model_types[2])
    comparison(feature_types[2], top_k_words, model_types[1])
    comparison(feature_types[2], top_k_words, model_types[2])

    # category特征表示方法，MyCategoricalNB模型准确率为: 0.9816727757414195
    # category特征表示方法，MyMultinomialNB模型准确率为: 0.9830056647784072
    # category特征表示方法，MyGaussianNB模型准确率为:0.9866711096301233
    # counts特征表示方法，MyMultinomialNB模型准确率为: 0.9836721092969011
    # counts特征表示方法，MyGaussianNB模型准确率为: 0.9860046651116294
    # tfidf特征表示方法，MyMultinomialNB模型准确率为: 0.984005331556148
    # tfidf特征表示方法，MyGaussianNB模型准确率为: 0.9850049983338887
