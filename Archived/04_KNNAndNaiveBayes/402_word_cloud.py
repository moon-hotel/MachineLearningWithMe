from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
import re


#
# word_cloud = WordCloud(background_color='white', max_font_size=70)
# top_k_words = {'a': 0.3, 'b': 0.89, 'c': 0.88, 'd': .67}
# word_cloud.fit_words(top_k_words)
# plt.imshow(word_cloud)
# plt.show()


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


def load_data_and_cut(file_path='./data/QuanSongCi.txt'):
    x_cut = []
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip('\n')
            seg_list = jieba.cut(clean_str(line), cut_all=False)
            tmp = " ".join(seg_list)
            x_cut.append(tmp)
    return x_cut


def get_tf_idf_rank(top_k_words=500, file_path=None):
    x = load_data_and_cut(file_path=file_path)
    tfidf = TfidfVectorizer(max_features=top_k_words)  # 默认值
    weight = tfidf.fit_transform(x).toarray()
    word = tfidf.get_feature_names()
    word_fre = {}
    for i in range(len(weight)):
        for j in range(len(word)):
            if word[j] not in word_fre:
                word_fre[word[j]] = weight[i][j]
            else:
                word_fre[word[j]] = max(word_fre[word[j]], weight[i][j])
    return word_fre


def show_word_cloud(word_fre):
    word_cloud = WordCloud(font_path='./data/simhei.ttf', background_color='white', max_font_size=70)
    word_cloud.fit_words(word_fre)
    plt.imshow(word_cloud)
    plt.xticks([])#去掉横坐标
    plt.yticks([])#去掉纵坐标
    plt.show()


if __name__ == '__main__':
    word_fre = get_tf_idf_rank(file_path='./data/QuanSongCi.txt')
    show_word_cloud(word_fre)

