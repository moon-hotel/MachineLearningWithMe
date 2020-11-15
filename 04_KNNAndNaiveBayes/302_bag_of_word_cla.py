from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.externals import joblib
import jieba
import re
import os


def clean_str(string, sep=" "):
    """
    该函数的作用是去掉一个字符串中的所有非中文字符
    :param string: 输入必须是字符串类型
    :param sep: 表示去掉的部分用什么填充，默认为一个空格
    :return: 返回处理后的字符串
    example:
    s = "祝你2018000国庆快乐！"
    print(clean_str(s))# 祝你 国庆快乐
    print(clean_str(s,sep=""))# 祝你国庆快乐
    """
    string = re.sub(r"[^\u4e00-\u9fff]", sep, string)
    string = re.sub(r"\s{1,}", sep, string)  # 若有空格，则最多只保留1个宽度
    return string.strip()


def load_data_and_cut(file_path='./data/ham_100.utf8'):
    x_cut = []
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip('\n')
            seg_list = jieba.cut(clean_str(line), cut_all=False)
            tmp = " ".join(seg_list)
            x_cut.append(tmp)
    return x_cut


def get_dataset(top_k_words=1000):
    x_pos = load_data_and_cut(file_path='./data/ham_5000.utf8')
    x_neg = load_data_and_cut(file_path='./data/spam_5000.utf8')
    y_pos = [1] * len(x_pos)
    y_neg = [0] * len(x_neg)
    x = x_pos + x_neg
    y = y_pos + y_neg
    X_train, X_test, y_train, y_test = \
        train_test_split(x, y, test_size=0.3, random_state=42)
    count_vec = CountVectorizer(max_features=top_k_words)
    X_train = count_vec.fit_transform(X_train)
    X_test = count_vec.transform(X_test)

    # print(x[:2])
    # print(len(count_vec.vocabulary_))
    return X_train, X_test, y_train, y_test


def train(X_train, X_test, y_train, y_test):
    model = MultinomialNB()
    model.fit(X_train, y_train)
    save_model(model)
    y_pre = model.predict(X_test)
    r = classification_report(y_test, y_pre)
    print(r)


def save_model(model, dir='MODEL'):
    if not os.path.exists(dir):
        os.mkdir(dir)
    joblib.dump(model, os.path.join(dir, 'model.pkl'))


def load_model(dir='MODEL'):
    path = os.path.join(dir, 'model.pkl')
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} 模型不存在，请先训练模型！")
    model = joblib.load(path)
    return model


def predict(X):
    model = load_model()
    y_pred = model.predict(X)
    print(y_pred)


if __name__ == '__main__':
    # x = load_data_and_cut()
    # print(x[:2])
    X_train, X_test, y_train, y_test = get_dataset()
    # train(X_train, X_test, y_train, y_test)
    predict(X_test)
