import jieba
from collections import Counter


def vetorization(s):
    cut_words = ""
    x_text = []
    # ---------分词处理------------------
    for item in s:
        seg_list = jieba.cut(item, cut_all=False)
        tmp = " ".join(seg_list)
        cut_words += (tmp + " ")
        x_text.append(tmp.split())
    all_words = cut_words.split()
    c = Counter()
    for x in all_words:
        if len(x) > 1 and x != '\r\n':
            c[x] += 1
    # ---------词频统计构造词表------------------
    vocab = []
    for (k, v) in c.most_common(8):  # 输出词频最高的前8个词
        vocab.append(k)
    # ---------向量化------------------
    x_vec = []
    for item in x_text:
        tmp = [0] * len(vocab)
        for i, w in enumerate(vocab):
            if w in item:
                tmp[i] = 1
        x_vec.append(tmp)
    print("词表：", vocab)
    print("文本：", x_text)
    print(x_vec)


if __name__ == '__main__':
    s = ['文本分词工具可用于对文本进行分词处理', '常见的用于处理文本的分词处理工具有很多']
    vetorization(s)
