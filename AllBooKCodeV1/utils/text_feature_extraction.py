from collections import Counter
import numpy as np


class VectWithoutFrequency(object):
    """

    """

    def __init__(self, top_k_words=500):
        self.top_k_words = top_k_words

    def _get_vocab(self, raw_documents):

        c = Counter()
        for sample in raw_documents:
            words_list = sample.split()
            for x in words_list:
                if len(x) > 1 and x != '\r\n':
                    c[x] += 1
        # ---------词频统计构造词表------------------
        vocab = []
        for (k, v) in c.most_common(self.top_k_words):  # 输出词频最高的前top_k_words个词
            vocab.append(k)
        return vocab

    def fit_transform(self, raw_documents):
        """
        拟合
        :param raw_documents: 原始样本，list， 每个元素为分词后的样本
        :return:
        """
        self.fit(raw_documents)
        x = self.transform(raw_documents)
        return x

    def transform(self, raw_documents):
        """
        :param raw_documents:
        :return:
        e.g.
        s = ['文本 分词 工具 可 用于 对 文本 进行 分词 处理', '常见 的 用于 处理 文本 的 分词 处理 工具 有 很多']
        vect = VectWithoutFrequency()
          x = vect.fit_transform(s)
          vect.vocab: ['文本', '分词', '处理', '工具', '用于', '进行', '常见', '很多']
        x:
          [[1, 1, 1, 1, 1, 1, 0, 0],
           [1, 1, 1, 1, 1, 0, 1, 1]]
        """
        x_vec = []
        for item in raw_documents:
            tmp = [0] * len(self.vocabulary)
            for i, w in enumerate(self.vocabulary):
                if w in item:
                    tmp[i] = 1
            x_vec.append(tmp)
        return np.array(x_vec)

    def fit(self, raw_documents):
        self.vocabulary = self._get_vocab(raw_documents)
