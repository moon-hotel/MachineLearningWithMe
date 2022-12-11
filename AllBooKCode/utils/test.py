"""
文件名: AllBooKCode/utils/test.py
创建时间: 2022/12/11 9:58 上午
作者: @空字符
公众号: @月来客栈
知乎: @月来客栈 https://www.zhihu.com/people/the_lastest
"""

from text_feature_extraction import VectWithoutFrequency


def test_VectWithoutFrequency():
    s = ['文本 分词 工具 可 用于 对 文本 进行 分词 处理', '常见 的 用于 处理 文本 的 分词 处理 工具 有 很多']
    vec = VectWithoutFrequency(8)
    print(vec.fit_transform(s))

    vec.fit(s)
    print(vec.transform(s))


if __name__ == '__main__':
    test_VectWithoutFrequency()
