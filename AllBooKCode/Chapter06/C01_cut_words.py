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
    s = "阿里通义千问正式推出了其最新版本的 Qwen Code v0.3.0，旨在进一步提升开发者的专业能力。这款专为 Qwen3-Coder 模型优化的命令行 AI 工作流工具，不仅具备强大的代码理解能力和自动化任务功能，还融入了智能辅助功能，让开发者在编程的过程中更加高效。"
    cutWords(s, cut_all=False)
    wordsCount(s)
