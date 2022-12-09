"""
文件名: C01_cut_words.py
创建时间:
作者: @空字符
公众号: @月来客栈
知乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

from collections import Counter
import jieba
import re


def cutWords(s, cut_all=False):
    cut_words = []
    s = re.sub("[A-Za-z0-9\：\·\—\，\。\“ \”]", "", s)
    seg_list = jieba.cut(s, cut_all=cut_all)
    cut_words.append("/".join(seg_list))
    print(cut_words)


def wordsCount(s):
    cut_words = ""
    s = re.sub("[A-Za-z0-9\：\·\—\，\。\“ \”]", "", s)
    seg_list = jieba.cut(s, cut_all=False)
    cut_words += (" ".join(seg_list))
    all_words = cut_words.split()
    c = Counter()
    for x in all_words:
        if len(x) > 1 and x != '\r\n':
            c[x] += 1
    vocab = []
    print('\n词频统计结果：')
    for (k, v) in c.most_common(5):  # 输出词频最高的前5个词
        print("%s:%d" % (k, v))
        vocab.append(k)
    print("词表：", vocab)


if __name__ == '__main__':
    s = "央视网消息：当地时间11日，美国国会参议院以88票对11票的结果通过了一项动议，允许国会“在总统以国家安全为由决定征收关税时”发挥一定的限制作用。这项动议主要针对加征钢铝关税的232调查，目前尚不具有约束力。动议的主要发起者——共和党参议员鲍勃·科克说，11日的投票只是一小步，他会继续推动进行有约束力的投票。"
    cutWords(s, cut_all=False)
    wordsCount(s)
