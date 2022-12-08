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
    """
    载入原始样本，分词并返回。
    返回后是一个List，list里的每个元素为一个样本，样本为分词后的结果
    :param file_path:
    :return:
    '本次 活动 只限 上海地区', '中信   国际   电子科技 有限公司 推出 新 产品',
     '您好   本 公司 主要 从事 税务代理   并 可代 开发票 范围 如下   商品销售 发票 ',
     ......]
    """
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
    y_pos, y_neg = [1] * len(x_pos), [0] * len(x_neg)
    x, y = x_pos + x_neg, y_pos + y_neg
    X_train, X_test, y_train, y_test = \
        train_test_split(x, y, test_size=0.3, random_state=42)
    count_vec = CountVectorizer(max_features=top_k_words)
    ## 考虑词频的词袋模型
    X_train = count_vec.fit_transform(X_train)
    X_test = count_vec.transform(X_test)
    # print(len(count_vec.vocabulary_)) # 输出词表长度
    return X_train, X_test, y_train, y_test


def train(X_train, X_test, y_train, y_test):
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    save_model(model)
    y_pre = model.predict(X_test)
    print("模型在测试集上的表现结果：")
    print(classification_report(y_test, y_pre))


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
    X_train, X_test, y_train, y_test = get_dataset()
    # train(X_train, X_test, y_train, y_test)
    predict(X_test)
