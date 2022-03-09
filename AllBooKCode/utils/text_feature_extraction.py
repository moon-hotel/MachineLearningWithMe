import jieba
from collections import Counter

import numpy as np


class VectWithoutFrequency(object):
    """
    s = ['文本分词工具可用于对文本进行分词处理', '常见的用于处理文本的分词处理工具有很多']
    vect = VectWithoutFrequency()
    x = vect.fit_transform(s)
    vect.vocab:
    ['文本', '分词', '处理', '工具', '用于', '进行', '常见', '很多']
    x:
    [[1, 1, 1, 1, 1, 1, 0, 0],
     [1, 1, 1, 1, 1, 0, 1, 1]]
    """

    def __init__(self, top_k_words=500):
        self.top_k_words = top_k_words

    @staticmethod
    def _cut_words(s):
        cut_words = ""
        x_text = []
        # ---------分词处理------------------
        for item in s:
            seg_list = jieba.cut(item, cut_all=False)
            tmp = " ".join(seg_list)
            cut_words += (tmp + " ")
            x_text.append(tmp.split())
        all_words = cut_words.split()
        return all_words, x_text

    def fit(self, s):
        return self.fit_transform(s)

    def transform(self, s):
        _, x_text = self._cut_words(s)
        x_vec = []
        for item in x_text:
            tmp = [0] * len(self.vocab)
            for i, w in enumerate(self.vocab):
                if w in item:
                    tmp[i] = 1
            x_vec.append(tmp)
        return np.array(x_vec)

    def fit_transform(self, s):
        all_words, x_text = self._cut_words(s)
        c = Counter()
        for x in all_words:
            if len(x) > 1 and x != '\r\n':
                c[x] += 1
        # ---------词频统计构造词表------------------
        self.vocab = []
        for (k, v) in c.most_common(self.top_k_words):  # 输出词频最高的前8个词
            self.vocab.append(k)
        # ---------向量化------------------
        x_vec = self.transform(s)
        return x_vec
