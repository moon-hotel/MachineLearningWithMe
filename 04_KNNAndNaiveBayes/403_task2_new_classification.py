import pandas as pd
import re
import jieba
import numpy as np


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
    string = re.sub(r"\s{2,}", sep, string)  # 若有空格，则最多只保留2个宽度
    return string.strip()


def cut_line(line):
    """
    该函数的作用是 先清洗字符串，然后分词
    :param line: 输入必须是字符串类型
    :return: 分词后的结果
    example:
    s ='我今天很高兴'
    print(cut_line(s))# 我 今天 很 高兴
    """
    line = clean_str(line)
    seg_list = jieba.cut(line)
    cut_words = " ".join(seg_list)
    return cut_words


def load_and_cut(data_dir=None):
    """
    该函数的作用是载入原始数据，然后返回处理后的数据
    :param data_dir:
    :return:
    content_seg=['经销商   电话   试驾   订车   憬 杭州 滨江区 江陵','计 有   日间 行 车灯 与 运动 保护 型']
    y = [1,1]
    """
    names = ['category', 'theme', 'URL', 'content']
    data = pd.read_csv(data_dir, names=names, encoding='utf8', sep='\t')
    data = data.dropna()  # 去掉所有含有缺失值的样本（行）
    content = data.content.values.tolist()
    content_seg = []
    for item in content:
        content_seg.append(cut_line(clean_str(item)))
    # labels = data.category.unique()
    label_mapping = {'汽车': 1, '财经': 2, '科技': 3, '健康': 4, '体育': 5, '教育': 6, '文化': 7, '军事': 8, '娱乐': 9, '时尚': 10}
    data['category'] = data['category'].map(label_mapping)
    y = np.array(data['category'])
    del data, content
    return content_seg, y


if __name__ == '__main__':
    data_dir = './data/sougounews/'
    x, y = load_and_cut(data_dir + 'train.txt')